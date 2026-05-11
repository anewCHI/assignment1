import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    交叉熵损失。
    """
    #提取维度信息
    vocab_size = logits.size(-1)
    
    m = torch.max(logits, dim=-1, keepdim=True).values
    
    #提取目标位置Logits
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    #计算Log-Sum-Exp项
    shifted_logits = logits - m

    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    
    #计算单个Token的损失
    loss = log_sum_exp - target_logits
    
    #返回整个Batch的平均值
    return torch.mean(loss)