import torch
import math
from collections.abc import Callable
from typing import Optional

#带有衰减逻辑的SGD
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                #获取状态字典来记录步数
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                
                t = state["t"]
                grad = p.grad.data
                

                p.data -= lr / math.sqrt(t + 1) * grad
                
                #更新步数
                state["t"] += 1
        return loss

#学习率调试实验
def run_experiment(learning_rate):
    print(f"\n--- Testing LR = {learning_rate} ---")
    #初始化权重
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=learning_rate)
    
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"Iter {t}: Loss = {loss.item():.4f}")
        loss.backward()
        opt.step()

#不同学习率测试
lrs_to_test = [1e1, 1e2, 1e3]
for lr in lrs_to_test:
    run_experiment(lr) 