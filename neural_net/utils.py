import os
import math
import random
import numpy as np
import torch

from torch import nn


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_prec_recall_f1_acc(conf_matr):
    """Per-class precision, recall, F1 and overall accuracy from a confusion
    matrix (rows = ground truth, columns = prediction)."""
    accuracy = np.trace(conf_matr) / conf_matr.sum()
    predicted_sum = conf_matr.sum(axis=0)
    gt_sum = conf_matr.sum(axis=1)
    diag = np.diag(conf_matr)
    precision = diag / predicted_sum.clip(min=1e-5)
    recall = diag / gt_sum.clip(min=1e-5)
    f1 = 2 * (precision * recall) / (precision + recall).clip(min=1e-5)
    return precision, recall, f1, accuracy


def warmup_cosine_lambda(warmup_epochs, max_epochs, min_lr_ratio=0.0):
    """LambdaLR multiplier: linear warmup to the base LR over `warmup_epochs`,
    then cosine decay to `min_lr_ratio` by `max_epochs`. Stepped once per epoch."""
    warmup_epochs = max(int(warmup_epochs), 0)

    def fn(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return fn


def initialize_weight(module):
    """Xavier-normal weights and zero bias; use via `model.apply(initialize_weight)`."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
