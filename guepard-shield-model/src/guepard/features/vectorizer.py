from typing import Iterator, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


class SyscallVectorizer:
    """
    Transforms token streams into Bag-of-Words (or n-grams) arrays.
    """

    def __init__(self, vocab_list=None, max_features=None, ngram_range=(1, 3)):
        self.vectorizer = CountVectorizer(
            vocabulary=vocab_list,
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
        )
        self.is_fitted = False

    def fit(self, corpus_stream: Iterator[List[str]], total: int | None = None):
        """
        Fits the vectorizer on a stream of token lists.
        """

        def _iter():
            for tokens in tqdm(
                corpus_stream, desc="Fitting vectorizer", unit="seq", total=total
            ):
                yield " ".join(tokens)

        self.vectorizer.fit(_iter())
        self.is_fitted = True

    def transform(self, tokens: List[str] | List[List[str]]) -> np.ndarray:
        """
        Transforms tokens to a dense feature array.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform.")

        if isinstance(tokens[0], list):
            texts = [" ".join(seq) for seq in tokens]
        else:
            texts = [" ".join(tokens)]  # type: ignore

        # We output float32 to be compatible with Keras / Surrogate datasets
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
