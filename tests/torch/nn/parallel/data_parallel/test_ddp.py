import torch.cuda
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from oslo.torch.distributed import ParallelContext

# parallel context 생성
from oslo.torch.nn.parallel.data_parallel.distributed_data_parallel import (
    DistributedDataParallel,
)

parallel_context = ParallelContext.from_torch(
    data_parallel_size=2,
    pipeline_parallel_size=1,
    tensor_parallel_size=1,
)

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 모델 생성 및 병렬화 수행
model_no_ddp = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2")).cuda()
model_ddp = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2")).cuda()
model_ddp = DistributedDataParallel(
    model_ddp,
    parallel_context=parallel_context,
)

# allocate_params(wrapper_ddp, parallel_context)
# allocate_params 함수는 추후에 모든 페러렐 래퍼를 관장하는 클래스에서 처리될 예정
# https://github.com/tunib-ai/oslo/blob/307131bbd5ed995ea8dca8ac541bfbce9bfec29b/oslo/pytorch/model_parallelism/model_parallel_engine.py

# 옵티마이저 생성
optimizer_no_ddp = Adam(model_no_ddp.parameters(), lr=3e-5)
optimizer_ddp = Adam(model_ddp.parameters(), lr=3e-5)

# 데이터셋 생성
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=2)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name="ddp")

# 학습 시작
for data in dataloader:
    optimizer_ddp.zero_grad()
    optimizer_no_ddp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_ddp = model_ddp(**inputs, labels=inputs["input_ids"]).loss
    loss_no_ddp = model_no_ddp(**inputs, labels=inputs["input_ids"]).loss

    if dist.get_rank() == 0:
        print(f"DDP:{loss_ddp}, NORMAL:{loss_no_ddp}")
        wandb.log({"DDP": loss_ddp, "NORMAL": loss_no_ddp})

    loss_ddp.backward()
    loss_no_ddp.backward()

    optimizer_ddp.step()
    optimizer_no_ddp.step()
