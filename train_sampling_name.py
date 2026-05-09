from torch.utils.data import DataLoader, DistributedSampler
import torch
import argparse
import datetime
import pdb
import time

import yaml
import os
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

# from evaluation.FID import calc_FID
# from evaluation.LPIPS import calc_LPIPS
from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file

# 重建数据集（需与训练一致）
train_dataset, val_dataset, test_dataset = get_dataset(config.data)

# 初始化采样器
train_sampler = DistributedSampler(train_dataset)

# 固定随机种子（需与训练相同）
torch.manual_seed(config.seed)  # 训练时的随机种子
sampled_indices_history = []

# 逐 epoch 恢复采样器的索引分配
for epoch in range(config.training.n_epochs):
    train_sampler.set_epoch(epoch)
    sampled_indices = train_sampler.indices  # 当前 epoch 的索引
    sampled_indices_history.append(sampled_indices)

    # 保存到文件（可选）
    with open(f"train_sampler_indices_epoch_{epoch}.txt", "w") as f:
        f.write("\n".join(map(str, sampled_indices)))
