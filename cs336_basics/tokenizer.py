import regex as re
from collections.abc import Iterable
class BPETokenizer:
    """
    BPE分词器实现
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        初始化分词器
        """
        #双向映射
        self.vocab = vocab  # ID到字节块
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()} #字节块到ID
        
        #合并规则转换为Rank字典
        self.merges = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        
        #特殊Token的正则表达式
        if self.special_tokens:
            #按照长度从长到短排序
            #优先匹配最长的特殊标记，防止重叠标记被错误拆分
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        #预分词正则表达式
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def encode(self, text: str) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。
        """
        if not text:
            return []
        if not self.special_regex:
            return self._encode_text_segment(text)
        tokens = []
        
        #记录上一次匹配结束的位置
        last_pos = 0
        
        for match in self.special_regex.finditer(text):
            pre_text = text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            special_tok = match.group()
            
            #特殊标记不参与合并
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])
            
            #更新标记
            last_pos = match.end()
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
        """
        对无特殊Token的纯文本片段合并
        """
        ids = []
        pre_tokens = self.gpt2_pat.findall(text)
        
        for p_tok in pre_tokens:
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]
            while len(byte_parts) >= 2:
                best_pair = None
                min_rank = float('inf')
                
                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                
                #找不到任何可以合并的规则时退出
                if best_pair is None:
                    break 
                
                #合并操作。
                new_byte_parts = []
                i = 0
                while i < len(byte_parts):
                    #合并成功
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2 # 跳过下一项
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_byte_parts #进入下一轮循环
            
            for part in byte_parts:
                ids.append(self.byte_to_id[part])
                
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        解码
        """
        byte_segments = [self.id_to_byte[i] for i in ids]
        
        full_bytes = b"".join(byte_segments)
        
        #解码为UTF-8字符串
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        迭代编码器
        """
        for chunk in iterable:
            #对每一块文本进行编码并通过yield输出结果
            yield from self.encode(chunk)