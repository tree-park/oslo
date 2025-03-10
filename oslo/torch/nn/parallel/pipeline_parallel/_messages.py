from dataclasses import dataclass
from typing import Optional, Any
from typing import Tuple, List, Union

import torch

MESSAGE_GENERATION = 0
REQUEST_GENERATION = 0


@dataclass(init=False)
class Message:
    comm_type: str
    # 1. request or response
    request_from: Optional[str]
    # 2. request module id
    exec_type: str
    # 3. forward or backward
    inputs: Optional[Any]
    # 4. input data for module execution
    outputs: Optional[Any]
    # 5. output data from module execution
    src_rank: int
    # 6. source pp rank
    dst_rank: int
    # 7. destination pp rank
    location: int
    # 8. The location of the module within the module graph
    in_autocast_context: bool
    # 9. Whether the requester is currently in a autocast context
    in_grad_related_context: bool
    # 10. Whether the requester is currently in a no grad/enable grad context
    use_activation_checkpointing: bool
    # 11. Whether activation checkpointing is enabled for the current module

    def __init__(self):
        global MESSAGE_GENERATION
        MESSAGE_GENERATION += 1
        self.tag = MESSAGE_GENERATION


@dataclass
class TensorStub(object):
    id: str
    dtype: torch.dtype
    shape: Union[List, Tuple]
    requires_grad: bool


@dataclass(init=False)
class RemoteWorkRequest:
    src: torch.device
    dst: torch.device
    location: str
    tag: int
    caller: str
    keys: tuple

    def __init__(self):
        global REQUEST_GENERATION
        REQUEST_GENERATION += 1
        self.tag = REQUEST_GENERATION


def generate_request(src, dst, location, caller, args, kwargs):
    req = RemoteWorkRequest()
    req.src = src
    req.dst = dst
    req.location = location
    req.caller = caller

    # merge kwargs into args
    keys, new_args = assemble_args(args, kwargs)
    req.keys = keys

    return req, new_args


def assemble_args(args, kwargs):
    new_args = []
    keys = []
    for v in args:
        if torch.is_tensor(v):
            v = v.contiguous()
        new_args.append(v)
        keys.append(None)

    for k, v in kwargs.items():
        if k is None:
            raise ValueError("None cannot be used the key of kwargs.")
        if torch.is_tensor(v):
            v = v.contiguous()
        new_args.append(v)
        keys.append(k)

    return tuple(keys), tuple(new_args)


def disassemble_new_args(new_args, keys):
    args = list()
    kwargs = dict()

    for k, v in zip(keys, new_args):
        if k is None:
            args.append(v)
        else:
            kwargs[k] = v

    return tuple(args), kwargs


def disassemble_result(result):
    if isinstance(result, torch.Tensor):
        args = (result,)
        kwargs = dict()
        wrapped = True
    elif isinstance(result, dict):
        args = tuple([])
        kwargs = result
        wrapped = False
    elif isinstance(result, (list, tuple)):
        args = tuple(result)
        kwargs = dict()
        wrapped = False
    else:
        raise NotImplementedError

    return args, kwargs, wrapped
