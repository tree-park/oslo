import torch
import torch.nn as nn


from transformers.utils import ExplicitEnum


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    ADAM = "adam"
    ADAMW = "adamw"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAFACTOR = "adafactor"
    ADAMW_BNB = "adamw_bnb_8bit"
    SGD = "sgd"
    NOVOGRAD = "novograd"
    LAMB = "lamb"


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
