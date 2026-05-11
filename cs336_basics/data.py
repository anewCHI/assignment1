import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在Numpy数组数据里集中随机采样一个批次。
    """
    #确定合法的最大起点索引
    n = len(dataset)
    max_idx = n - context_length - 1
    
    #随机产生起始位置（个数为batch_size）
    ix = torch.randint(0, max_idx + 1, (batch_size,))
    
    #根据索引提取输入和目标
    x_stack = [dataset[i : i + context_length] for i in ix]
    y_stack = [dataset[i + 1 : i + context_length + 1] for i in ix]
    
    #转换为张量并移动到指定设备
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()
    
    return x, y