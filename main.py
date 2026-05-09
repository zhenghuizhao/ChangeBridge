# -*- coding: utf-8 -*-
"""
启动脚本：支持 CPU / 单卡 / 多卡(DDP) 的训练与测试流程。

主要功能：
- 解析命令行参数并与 YAML 配置合并；
- 设置可复现随机种子；
- 根据 --gpu_ids 路由到 CPU/单卡/多卡(DDP) 执行路径；
- DDP 模式下，即使出现异常也会在 finally 中销毁进程组，防止僵尸进程与端口占用。
"""

import warnings
# 屏蔽特定的 PyTorch 警告（与旧版权重 TypedStorage 相关），避免日志干扰
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np

from utils import dict2namespace, get_runner, namespace2dict
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_args_and_config():
    """
    解析命令行参数，加载 YAML 配置，并应用运行时覆盖项。

    返回
    ----
    namespace_config : 类 Namespace 的对象
        从 dict 转换而来，便于点操作访问；附带 .args（命令行参数）。
    dict_config : dict
        应用所有覆盖项后的纯字典视图。

    运行时覆盖规则
    ------------
    - --resume_model / --resume_optim：写入 config.model.*
    - --max_epoch / --max_steps：覆盖 training.n_epochs / training.n_steps
    """
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # === Parser help text in ENGLISH ===
    parser.add_argument('-c', '--config', type=str, default='BB_base.yml',
                        help='Path to the YAML config file')
    parser.add_argument('-s', '--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results',
                        help='Directory to save results (if used by runner)')

    parser.add_argument('-t', '--train', action='store_true', default=False,
                        help='Run training; otherwise run testing')
    parser.add_argument('--sample_to_eval', action='store_true', default=False,
                        help='Sample for evaluation (runner-specific feature)')
    parser.add_argument('--sample_at_start', action='store_true', default=False,
                        help='Sample at start (for debugging)')
    parser.add_argument('--save_top', action='store_true', default=False,
                        help='Save checkpoint when top metric improves')

    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='Comma-separated GPU ids like "0,1,2,3"; use "-1" for CPU')
    parser.add_argument('--port', type=str, default='12355',
                        help='DDP master port (single-machine multi-GPU)')

    parser.add_argument('--resume_model', type=str, default=None,
                        help='Path to model checkpoint to resume')
    parser.add_argument('--resume_optim', type=str, default=None,
                        help='Path to optimizer/scheduler checkpoint to resume')

    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Override training.n_epochs')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Override training.n_steps')
    # === END parser (English only) ===

    args = parser.parse_args()

    # 载入 YAML 配置
    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    # dict -> namespace，便于点式访问
    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args  # 附加命令行参数

    # 运行时覆盖
    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    # 同时返回 dict 视图（部分下游可能需要）
    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def set_random_seed(SEED: int = 1234) -> None:
    """
    统一设置随机种子，尽量提升运行的可复现性。

    参数
    ----
    SEED : int
        基础随机种子。

    说明
    ----
    - 将 cuDNN 置为 deterministic，以减少非确定性；
    - 可复现性通常会降低某些性能并限制部分算子。
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def DDP_run_fn(rank: int, world_size: int, config) -> None:
    """
    多卡（DDP）模式下由每个子进程执行的入口函数。

    参数
    ----
    rank : int
        当前进程的本地 rank（范围 [0, world_size-1]）。
    world_size : int
        进程总数（通常等于可见 GPU 数）。
    config : Namespace-like
        配置对象（会被深拷贝传入每个子进程）。

    行为
    ----
    1) 初始化 DDP 通信（NCCL 后端，单机使用 localhost/port）；
    2) 根据 rank 绑定对应 GPU，并把 device/local_rank 写回 config；
    3) 构建 runner，按 --train 调用 train() 或 test()；
    4) finally 中无论是否异常都销毁进程组，确保环境干净。
    """
    # 单机默认设置；如需多机，请改为外部 launch（torchrun 等）传入
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port

    # 初始化进程组（NCCL）
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    try:
        # 每个进程内设置随机种子
        set_random_seed(config.args.seed)

        # 将本进程映射到对应的本地 CUDA 设备
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

        # 记录设备信息到配置，便于下游使用
        config.training.device = [torch.device(f"cuda:{local_rank}")]
        config.training.local_rank = local_rank
        print(f'[DDP] rank={rank} using device: {config.training.device}')

        # 构建 runner 并执行
        runner = get_runner(config.runner, config)
        if config.args.train:
            runner.train()
        else:
            with torch.no_grad():
                runner.test()

    finally:
        # 无论是否报错，都要销毁进程组，避免端口占用/僵尸进程
        if dist.is_initialized():
            print(f"[DDP] Destroying process group for rank {rank}")
            dist.destroy_process_group()


def CPU_singleGPU_launcher(config) -> None:
    """
    CPU 或 单卡 GPU 的单进程执行入口。

    参数
    ----
    config : Namespace-like
        完整配置对象；需包含 .args.train 与 .training.device。

    行为
    ----
    - 设置随机种子；
    - 构建 runner；
    - 根据 --train 选择 train()/test()。
    """
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()


def DDP_launcher(world_size: int, run_fn, config) -> None:
    """
    使用 torch.multiprocessing.spawn 启动 DDP，按 world_size 生成子进程。

    参数
    ----
    world_size : int
        进程数量（通常与使用的 GPU 数一致）。
    run_fn : Callable
        子进程执行的函数（如 DDP_run_fn）。
    config : Any
        配置对象；会被深拷贝传给每个子进程。

    注意
    ----
    - 调用之前务必正确设置 CUDA_VISIBLE_DEVICES；
    - spawn 启动方式跨平台更稳健。
    """
    mp.spawn(
        run_fn,
        args=(world_size, copy.deepcopy(config)),
        nprocs=world_size,
        join=True
    )


def main() -> None:
    """
    程序入口：
    1) 解析/合并配置；
    2) 根据 --gpu_ids 路由到 CPU / 单卡 / 多卡(DDP) 执行路径。
    """
    nconfig, _ = parse_args_and_config()
    args = nconfig.args

    gpu_ids = args.gpu_ids

    if gpu_ids == "-1":
        # CPU 路径
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            # 单机多卡 DDP
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            # 单卡路径
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)


if __name__ == "__main__":
    main()
