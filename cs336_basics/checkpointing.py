import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    储存当前训练状态。
    """
    # 1. 创建一个字典来保存模型权重、优化器状态和迭代次数
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 2. 将字典保存到指定位置
    torch.save(checkpoint, out)

def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点拉取状态进行恢复，同时返回已保存的迭代次数。
    """
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']