import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json


def train_bpe(
    input_path: str | os.PathLike,  #语料文件的路径
    vocab_size: int,             #目标词表大小
    special_tokens: list[str],   #要保留的特殊Token
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练BPE
    """
    
    #初始化基础词表
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    
    #语料读取与分割
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    """
    特殊token处理
    """
    if special_tokens:
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        parts = re.split(f"({special_regex})", text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    #预分词
    #GPT-2的正则表达式
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
            
    #快速合并
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)
    stats = defaultdict(int)
    
    #倒排索引进行性能优化
    indices = defaultdict(set)
    
    #初始化 
    for idx, word in enumerate(words_list):
        freq = counts_list[idx] #获取频率
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] += freq          #累加该pair的全局频率
            indices[pair].add(idx)       #加入该pair对应的倒排列表
            
    merges = [] #存储合并规则

    #迭代合并流程
    #循环执行
    for _ in range(num_merges):
        #停止条件
        if not stats:
            break

        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        if stats[best_pair] <= 0:
            break
            
        #记录合并
        merges.append(best_pair)
        #创建新Token
        new_token = best_pair[0] + best_pair[1]

        relevant_indices = list(indices[best_pair])
        

        for idx in relevant_indices:
            word = words_list[idx] #获取单词
            freq = counts_list[idx] #获取频率
            
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    
                    #频率更新
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]

                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                      

                    word[i] = new_token     #将第一个字节替换为新Token
                    del word[i+1]           #删除第二个字节
                    
                    #添加新产生的Pair的频率与索引
                    if i > 0:
                        new_prev = (word[i-1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)

                    if i < len(word) - 1:
                        new_next = (word[i], word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)

                else:
                    i += 1
        
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    #最终词表

    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
        
    # 添加特殊token到词表末尾
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes

    return vocab, merges


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    #映射表初始化
    byte_encoder = bytes_to_unicode()

    #词表保存
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    #合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/owt_train.txt'
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    special_tokens = ['<|endoftext|>']
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'data/owt_out'
    print(f'Start {vocab_size}...')
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()