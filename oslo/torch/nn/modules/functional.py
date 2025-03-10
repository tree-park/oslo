import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.functional import (
    _in_projection,
    _in_projection_packed,
    _mha_shape_check,
    _scaled_dot_product_attention,
    pad,
)

from oslo.torch._C import get_softmax_kernel
from oslo.torch.distributed._seed.helper import seed
from oslo.torch.distributed.nn.functional import ring_av, ring_qk
from oslo.torch.distributed.parallel_mode import ParallelMode

"""
Autograd Functions
"""
global fused_layer_norm_cuda
fused_layer_norm_cuda = None

global ngram_repeat_block_cuda
ngram_repeat_block_cuda = None


# Utils from apex
def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())


@torch.jit.script
def _fused_gelu_fwb(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


@torch.jit.script
def _fused_bias_gelu_fwb(y, bias):
    x = y + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_bias_gelu_bwd(g, y, bias):
    x = y + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class _FusedGeLUFunction(torch.autograd.Function):
    """
    Kernel fusion function: GeLU
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return _fused_gelu_fwb(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = _fused_gelu_bwd(grad_output, input)
        return tmp, tmp


class _FusedBiasGeLUFunction(torch.autograd.Function):
    """
    Kernel fusion function: Bias + GeLU
    """

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return _fused_bias_gelu_fwb(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = _fused_bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


class _FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            from oslo.torch._C import FusedLayerNormBinder

            fused_layer_norm_cuda = FusedLayerNormBinder().bind()
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.layer_norm_forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        (
            grad_input,
            grad_weight,
            grad_bias,
        ) = fused_layer_norm_cuda.layer_norm_backward_affine(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            bias_,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


class _FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            from oslo.torch._C import FusedLayerNormBinder

            fused_layer_norm_cuda = FusedLayerNormBinder().bind()
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_norm_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_norm_backward_affine(
            grad_output.contiguous(),
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            ctx.eps,
        )
        return grad_input, grad_weight, None, None


class _FusedLayerNormAffineMixedDtypesFunction(_FusedLayerNormAffineFunction):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            from oslo.torch._C import FusedLayerNormBinder

            fused_layer_norm_cuda = FusedLayerNormBinder().bind()
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        (
            output,
            mean,
            invvar,
        ) = fused_layer_norm_cuda.layer_norm_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output


class _FusedRMSNormAffineMixedDtypesFunction(_FusedRMSNormAffineFunction):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            from oslo.torch._C import FusedLayerNormBinder

            fused_layer_norm_cuda = FusedLayerNormBinder().bind()
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_norm_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, ctx.eps
        )

        ctx.save_for_backward(input_, weight_, invvar)
        return output


class _FusedLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            from oslo.torch._C import FusedLayerNormBinder

            fused_layer_norm_cuda = FusedLayerNormBinder().bind()
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.layer_norm_forward(
            input_, ctx.normalized_shape, ctx.eps
        )
        ctx.save_for_backward(input_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, mean, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.layer_norm_backward(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            ctx.eps,
        )
        return grad_input, None, None


class _FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            from oslo.torch._C import FusedLayerNormBinder

            fused_layer_norm_cuda = FusedLayerNormBinder().bind()
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_norm_forward(
            input_, ctx.normalized_shape, ctx.eps
        )
        ctx.save_for_backward(input_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.rms_norm_backward(
            grad_output.contiguous(), invvar, input_, ctx.normalized_shape, ctx.eps
        )
        return grad_input, None, None


class _FusedScaleUpperTriangMaskSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = (
            get_softmax_kernel().fused_scaled_upper_triang_masked_softmax_forward(
                inputs, scale_t[0]
            )
        )

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = (
            get_softmax_kernel().fused_scaled_upper_triang_masked_softmax_backward(
                output_grads, softmax_results, scale_t[0]
            )
        )

        return input_grads, None


class _FusedScaleMaskSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])

        softmax_results = get_softmax_kernel().fused_scaled_masked_softmax_forward(
            inputs, mask, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = get_softmax_kernel().fused_scaled_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class _NGramRepeatBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size):
        global ngram_repeat_block_cuda
        if ngram_repeat_block_cuda is None:
            from oslo.torch._C import NgramRepeatBlockBinder

            ngram_repeat_block_cuda = NgramRepeatBlockBinder().bind()
        return ngram_repeat_block_cuda.forward(
            tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size
        )

    def backward(*args):
        raise NotImplementedError


"""
User Functions
"""


def fused_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormAffineFunction.apply(*args)


def fused_layer_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormFunction.apply(*args)


def mixed_dtype_fused_layer_norm_affine(
    input, weight, bias, normalized_shape, eps=1e-6
):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormAffineMixedDtypesFunction.apply(*args)


def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormAffineFunction.apply(*args)


def fused_rms_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormFunction.apply(*args)


def mixed_dtype_fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormAffineMixedDtypesFunction.apply(*args)


def fused_gelu(x):
    return _FusedGeLUFunction.apply(x)


def fused_bias_gelu(x, bias):
    return _FusedBiasGeLUFunction.apply(x, bias)


@torch.jit.script
def fused_bias_dropout(x, bias, p, training, inplace):
    # type: (Tensor, Tensor, float, bool, bool) -> Tensor
    return F.dropout(x + bias, p=p, training=training, inplace=inplace)


def _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32):
    assert input.dim() == 4, "input must be be `(batch, nhead, len_q, len_k)`."
    assert scale is not None, "scale must not be None."
    assert scale == 1.0 or softmax_in_fp32, "softmax should be in fp32 when scaled"


def _is_fused_scale_mask_softmax_available(
    input, scale, softmax_in_fp32, use_triang_mask
):
    bsz, np, sq, sk = input.size()
    dtype = input.dtype
    _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32)

    if dtype != torch.float16 and dtype != torch.bfloat16:
        return False

    if sk > 4096 or sk <= 0:
        return False

    if softmax_in_fp32 is True:
        return False

    bsz_per_block = get_softmax_kernel().get_batch_per_block(sq, sk, bsz, np)

    if use_triang_mask:
        if sq == sk and (sk <= 64 or sk % 4 == 0) and (bsz * np) % bsz_per_block == 0:
            return True
    else:
        if sq > 1 and sq % bsz_per_block == 0:
            return True

    return False


def _fused_scale_mask_softmax_torch(input, scale, softmax_in_fp32, mask):
    original_input_dtype = input.dtype
    _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32)

    if softmax_in_fp32 and original_input_dtype != torch.float32:
        input = input.float()

    input = input * scale
    mask_output = input.masked_fill_(mask, -10000.0) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)

    if softmax_in_fp32 and original_input_dtype != torch.float32:
        if original_input_dtype == torch.float16:
            probs = probs.half()
        else:
            probs = probs.bfloat16()

    return probs


def _fused_scale_mask_softmax_cuda(input, scale, use_triang_mask, pad_mask):
    bsz, np, sq, sk = input.size()
    _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32=False)

    if use_triang_mask:
        if pad_mask is not None:
            input += pad_mask
        output = _FusedScaleUpperTriangMaskSoftmaxFunction.apply(
            input.view(-1, sq, sk),
            scale,
        )
        return output.view(bsz, np, sq, sk)
    else:
        if pad_mask is not None:
            if pad_mask.size(2) == 1:
                pad_mask = pad_mask.repeat(1, 1, sq, 1)
            return _FusedScaleMaskSoftmaxFunction.apply(
                input,
                pad_mask.bool(),
                scale,
            )
        else:
            pad_mask = torch.zeros(1, 1, sq, sk, device=input.device, dtype=input.dtype)
            return _FusedScaleMaskSoftmaxFunction.apply(
                input,
                pad_mask.bool(),
                scale,
            )


def fused_scale_mask_softmax(
    input, scale, use_triang_mask, softmax_in_fp32, pad_mask=None
):
    scale = scale if scale is not None else 1.0
    if _is_fused_scale_mask_softmax_available(
        input, scale, softmax_in_fp32, use_triang_mask
    ):
        return _fused_scale_mask_softmax_cuda(input, scale, use_triang_mask, pad_mask)
    else:
        return _fused_scale_mask_softmax_torch(input, scale, softmax_in_fp32, pad_mask)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    use_sequence_parallel: bool = False,
    parallel_context: object = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
        use_sequence_parallel: If true, use SequenceParallel. Default: False
        parallel_context: global parallel context.

    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # SP
    if use_sequence_parallel:

        # [batch_size, num_heads, sub_seq_len, seq_len]
        attn_scores = ring_qk(
            sub_q=q,
            sub_k=k,
            parallel_context=parallel_context,
        )

        attn_scores /= math.sqrt(head_dim)

        # TODO: apply ScaleMaskSoftmax
        # TODO: test attn_mask
        # attn_output_weights = FusedScaleMaskSoftmax(attn_scores, attn_mask)
        attn_output_weights = torch.softmax(attn_scores, dim=-1)

        with seed(ParallelMode.TENSOR):
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = ring_av(
            sub_attn=attn_output_weights,
            sub_v=v,
            parallel_context=parallel_context,
        )

        # change view [batch_size, num_heads, sub_seq_len, head_size]
        output_size = (bsz, num_heads, tgt_len, head_dim)
        attn_output = attn_output.view(*output_size)

        # [batch_size, num_heads, sub_seq_len, head_size] -> [sub_seq_len, batch_size, num_heads, head_size]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()

        # [sub_seq_len, batch_size, num_heads, head_size] -> [sub_seq_len, batch_size, hidden_size]
        new_attn_output_shape = attn_output.size()[:-2] + (head_dim * num_heads,)
        attn_output = attn_output.view(*new_attn_output_shape)

        # linear projection
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        # change target length "query.shape[0]" to "sub_seq_len"
        tgt_len = attn_scores.size(-2)
        # change target length "k.size(1)" to "seq_len"
        src_len = attn_scores.size(-1)
    else:
        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = _scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p
        )
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights and not use_sequence_parallel:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
