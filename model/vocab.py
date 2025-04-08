import json
import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.build_done = False

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, text):
        # Simple word tokenizer (modify as needed)
        return re.findall(r"\w+", text.lower())

    def build_vocab(self, caption_list):
        counter = Counter()
        for caption in caption_list:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        # Add special tokens first
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
        idx = len(SPECIAL_TOKENS)

        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.build_done = True

    def numericalize(self, caption, max_len=20):
        tokens = self.tokenize(caption)
        ids = [self.word2idx.get("<start>")]
        for token in tokens:
            ids.append(self.word2idx.get(token, self.word2idx["<unk>"]))
        ids.append(self.word2idx.get("<end>"))

        if len(ids) < max_len:
            ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids
