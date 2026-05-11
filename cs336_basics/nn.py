import torch
import torch.nn as nn
import math
from einops import rearrange

"""
全连接层
"""

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        #定义权重W
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        #初始化权重
        std = (2.0 / (in_features + out_features)) ** 0.5
        #在[-3sigma, 3sigma]截断
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i, oi -> ...o', x, self.weight)



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        #必须初始化为全1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        #转换为float32，防止计算溢出
        x_float = x.to(torch.float32)

        #均方根计算
        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        #归一化，乘可学习的增益参数
        result = (x_float / rms) * self.weight

        #转换为原始类型
        return result.to(in_dtype)
        

def silu_fn(in_features):

    return in_features * torch.sigmoid(in_features)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        return self.w2(gate * signal)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块
        """
        super().__init__()
        self.d_k = d_k
        
        #计算频率
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 形状: (d_k/2,)
        
        #位置序列
        t = torch.arange(max_seq_len, device=device).float()
        
        #所有位置的所有角度的计算
        freqs_matrix = torch.outer(t, freqs)
        
        #预计算cos和sin
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #提取cos/sin
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        #维度对齐
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        #拆分，旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
    

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        #指数计算
        exp_x = torch.exp(x_stable)
        
        #分母的各指数之和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        #最终结果
        return exp_x / sum_exp

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    d_k = Q.size(-1)

    scores = torch.einsum('...nk, ...mk-> ...nm', Q, K) / math.sqrt(d_k)
    
    #应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    #Softmax归一化
    probs = softmax(scores, dim=-1)
    
    #Value的加权求和
    output = torch.einsum('...nm, ...mk-> ...nk', probs, V)
    
    return output

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = False, 
                 max_seq_len=None, theta=None, 
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        #Q,K,V的投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        #定义输出投影
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        #实例化RoPE
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b, s, d = x.shape

        q = rearrange(self.q_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        

        #应用RoPE
        if self.rope is not None:
            if token_positions is None:
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s, device=x.device).expand(*batch_dims, s)
            
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        #下三角矩阵
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        #SDPA
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        #合并与输出投影
        attn_out = rearrange(attn_out, '... h s d -> ... s (h d)')
        return self.output_proj(attn_out)


import torch
import torch.nn as nn
from .nn import Embedding, RMSNorm, Linear, CausalSelfAttention, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int,
                 theta: float, device=None, dtype=None, 
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"
                 ):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        #初始化 Attention
        self.attn = CausalSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=context_length, 
            theta=theta,
            device=device, 
            dtype=dtype
        )

        #初始化Norm层
        if use_rms_norm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        #初始化FFN
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            d_ff = 4 * d_model
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        #Pre-norm
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        
        #Post-norm
        elif self.norm_mode == "post":
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
            
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float, 
                 device=None, dtype=None,
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"):
        super().__init__()
        self.context_length = context_length
        
        #Token Embedding层
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        #Transformer Blocks堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta, 
                device=device, dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])
        
        #最终输出层
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()


            
        #Linear层映射回词表大小
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        b, s = token_ids.shape
        
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)
        
        #Embedding
        x = self.token_embeddings(token_ids)
        
        #逐层通过Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
            
        #最终归一化
        x = self.ln_final(x)
        
        #投影到词表空间，得到logits
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int, 
        eos_token_id: int = None, 
        temperature: float = 1.0, 
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        从模型生成文本 ID 序列。
        """
        #评估模式
        self.eval()
        
        #拷贝输入，避免修改原始数据
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            #裁剪输入
            idx_cond = generated[:, -self.context_length:]
            
            #前向传播
            logits = self.forward(idx_cond) # (Batch, T, Vocab)
            logits = logits[:, -1, :]      # (Batch, Vocab)
            
            #应用温度
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8)
            
            #应用Top-P过滤
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)
            
            #归一化并采样
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            
            #新词拼接
            generated = torch.cat((generated, next_token), dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        #累积概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        
        #确保至少保留第一个最高概率词，
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        #设置移除的token分数为负无穷
        #用scatter将掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits