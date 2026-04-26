import os
import re
import json
from collections import defaultdict, Counter
import regex as re  # 保持与GPT2一致的正则引擎

def bytes_to_unicode():
    """
    精确复刻GPT-2的bytes_to_unicode映射，确保与测试用的参考数据兼容
    """
    # 可见ASCII字符
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    # 补充不可见字节的映射
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 转换为字符
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,  # 兼容测试传入的额外参数
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    修正后的BPE训练逻辑，严格匹配测试要求
    """
    # 1. 初始化基础词表（0-255字节）
    vocab = {i: bytes([i]) for i in range(256)}
    num_special = len(special_tokens)
    num_merges = vocab_size - 256 - num_special
    
    # 边界检查：避免无效的合并次数
    if num_merges < 0:
        raise ValueError(f"vocab_size({vocab_size}) 小于基础字节数(256) + 特殊token数({num_special})")

    # 2. 读取并处理语料（隔离特殊token）
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 分割特殊token，确保其不参与BPE训练
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(f"({special_pattern})", text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    # 3. GPT2风格预分词
    # 精确复用GPT2的预分词正则表达式
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # 统计预分词后的词频（字节元组形式）
    raw_counts = Counter()
    for segment in train_segments:
        # 过滤空片段
        if not segment.strip():
            continue
        words = gpt2_pat.findall(segment)
        for word in words:
            # 转换为字节元组（不可变类型，用于Counter）
            byte_tokens = tuple(bytes([b]) for b in word.encode("utf-8"))
            raw_counts[byte_tokens] += 1

    # 4. 构建高效数据结构
    # 转换为可变列表（用于后续合并修改）
    words_list = [list(word) for word in raw_counts.keys()]
    counts_list = [raw_counts[word] for word in raw_counts.keys()]
    
    # 初始化字节对统计和倒排索引
    stats = defaultdict(int)
    indices = defaultdict(set)  # 记录每个字节对出现在哪些单词中

    for word_idx, word in enumerate(words_list):
        freq = counts_list[word_idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(word_idx)

    merges = []

    # 5. 核心合并逻辑（严格按频率+字典序）
    for _ in range(num_merges):
        if not stats:
            break  # 无更多可合并的字节对
        
        # 选择最佳字节对：频率最高 → 字典序最大（与测试参考一致）
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        if stats[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]

        # 获取所有包含该字节对的单词索引（复制避免迭代时修改）
        relevant_word_indices = list(indices[best_pair])
        
        # 更新受影响的单词、统计和索引
        for word_idx in relevant_word_indices:
            word = words_list[word_idx]
            freq = counts_list[word_idx]
            i = 0
            
            while i < len(word) - 1:
                if (word[i], word[i+1]) == best_pair:
                    # 1. 移除旧的相邻字节对统计
                    # 左邻居
                    if i > 0:
                        left_pair = (word[i-1], word[i])
                        stats[left_pair] -= freq
                        if stats[left_pair] == 0:
                            del stats[left_pair]
                    # 右邻居
                    if i < len(word) - 2:
                        right_pair = (word[i+1], word[i+2])
                        stats[right_pair] -= freq
                        if stats[right_pair] == 0:
                            del stats[right_pair]

                    # 2. 执行合并
                    word[i] = new_token
                    del word[i+1]

                    # 3. 添加新的相邻字节对统计
                    # 左邻居
                    if i > 0:
                        new_left_pair = (word[i-1], word[i])
                        stats[new_left_pair] += freq
                        indices[new_left_pair].add(word_idx)
                    # 右邻居
                    if i < len(word) - 1:
                        new_right_pair = (word[i], word[i+1])
                        stats[new_right_pair] += freq
                        indices[new_right_pair].add(word_idx)

                    # 合并后不移动索引（处理连续相同对）
                else:
                    i += 1

        # 清理已合并的字节对
        del stats[best_pair]
        del indices[best_pair]

    # 6. 构建最终词表
    # 添加合并生成的token（ID从256开始）
    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]
    # 添加特殊token
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 确保词表大小符合要求（截断或补全）
    if len(vocab) > vocab_size:
        # 截断超出的token（保留前vocab_size个）
        vocab = {k: v for k, v in list(vocab.items())[:vocab_size]}
    elif len(vocab) < vocab_size:
        # 补充空token（应对合并次数不足的情况）
        while len(vocab) < vocab_size:
            vocab[len(vocab)] = b""

    return vocab, merges

def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    byte_encoder = bytes_to_unicode()

    # 保存vocab.json
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 保存merges.txt
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" # 原始文本路径
    vocab_size = 10000 # 作业要求的词表大小
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_tokenizer_files(vocab, merges, output_dir)
    print(f"训练完成：词表大小={len(vocab)}, 合并规则数={len(merges)}")

if __name__ == "__main__":
    main()