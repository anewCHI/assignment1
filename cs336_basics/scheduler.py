import math

def get_lr_cosine_schedule(
    it: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int
) -> float:
    """
    计算带预热的余弦退火学习率。
    """
    
    #预热阶段(线性增长)
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    #退火后阶段（保持最小学习率）
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    #余弦退火阶段
    #退火阶段的进度
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    #余弦系数
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    #最终学习率
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)