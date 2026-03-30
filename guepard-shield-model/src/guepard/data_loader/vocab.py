import json
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

from tqdm import tqdm

from ..config import VocabConfig


class SyscallVocab:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"

    def __init__(self, config: Optional[VocabConfig] = None):
        if config is None:
            config = VocabConfig()
        self.config = config
        self.token2id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
        }
        self.id2token = {v: k for k, v in self.token2id.items()}

    def build(self, corpus_stream, total: Optional[int] = None):
        """Build vocabulary from a stream of sequences of tokens."""
        freqs = Counter()
        for seq in tqdm(corpus_stream, desc="Building vocab", unit="seq", total=total):
            freqs.update(seq)

        sorted_tokens = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))

        for token, count in sorted_tokens:
            if count >= self.config.min_freq:
                if len(self.token2id) < self.config.max_size:
                    idx = len(self.token2id)
                    self.token2id[token] = idx
                    self.id2token[idx] = token
                else:
                    break

    def encode(self, seq: list[str]) -> list[int]:
        unk_id = self.token2id[self.UNK_TOKEN]
        return [self.token2id.get(token, unk_id) for token in seq]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(idx, self.UNK_TOKEN) for idx in ids]

    def save(self, path: Path | str):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)

    @classmethod
    def from_corpus(
        cls, corpus_stream: Iterator[List[str]], config: Optional[VocabConfig] = None
    ) -> "SyscallVocab":
        vocab = cls(config)
        vocab.build(corpus_stream)
        return vocab

    @classmethod
    def load(
        cls, path: Path | str, config: Optional[VocabConfig] = None
    ) -> "SyscallVocab":
        vocab = cls(config)
        with open(path, "r") as f:
            data = json.load(f)
        vocab.token2id = data["token2id"]
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        return vocab

    def __len__(self):
        return len(self.token2id)
