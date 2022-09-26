import math
import os
import random
import re
import shutil
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import logging
from tqdm.auto import tqdm
import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PretrainedConfig,
    __version__,
)
from transformers.utils import (
    find_labels,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from oslo.torch.optim import ZeroRedundancyOptimizer
from oslo.torch.nn.parallel.utils import allocate_params
from oslo.torch.optim.sharded_grad_scaler import ShardedGradScaler
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
    ShardedDataParallel,
    FullyShardedDataParallel,
    DistributedDataParallel,
    SequenceDataParallel,
)
from .data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from .training_args import TrainingArguments
from .trainer_utils import (
    unwrap_model,
    OptimizerNames,
)

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.yaml"

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


class Trainer:
    def __init__(
            self,
            model: nn.Module = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):

        if args is None:
            # No Arguments passed
            output_dir = "tmp_trainer"
            logging.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)

        self.args = args

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.optimizer = None
        self.lr_scheduler = None
        self.parallel_context = None
        self.model_wrappers = []
        if args.oslo_config:
            self.parallel_context, self.model_wrappers = (
                args.parallel_context,
                args.model_wrappers,
            )

        default_collator = (
            default_data_collator
            if tokenizer is None
            else DataCollatorWithPadding(tokenizer)
        )
        self.data_collator = (
            data_collator if data_collator is not None else default_collator
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # Define and add callback
        # default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        default_callbacks = DEFAULT_CALLBACKS
        callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.callback_handler.add_callback(DEFAULT_PROGRESS_CALLBACK)

        # TODO Grade Scaler
        # TODO Label Smoother

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()
        default_label_names = find_labels(self.model.__class__)
        logging.info(f"default_label_names: {default_label_names}")
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        return self.args.process_index == 0