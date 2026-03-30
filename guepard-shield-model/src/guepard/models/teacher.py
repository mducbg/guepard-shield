import keras

from ..config import TeacherConfig


def _compute_loss(y, y_pred, temperature):
    """Shared soft/hard label loss for teacher models."""
    y = keras.ops.convert_to_tensor(y)
    if keras.backend.is_float_dtype(y.dtype):
        soft_pred = keras.ops.softmax(y_pred / temperature)
        return keras.ops.mean(keras.losses.categorical_crossentropy(y, soft_pred))
    return keras.ops.mean(
        keras.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True)
    )


class SyscallLSTM(keras.Model):
    """
    Bidirectional LSTM teacher model. Fewer parameters than Transformer,
    better inductive bias for sequential syscall data on small datasets.
    """

    def __init__(self, config: TeacherConfig, **kwargs):
        super().__init__(**kwargs)
        self.temperature = config.temperature
        # mask_zero=True propagates padding mask to LSTM
        self.embedding = keras.layers.Embedding(
            config.vocab_size, config.d_model, mask_zero=True
        )
        self.lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(config.d_model // 2, dropout=config.dropout)
        )
        self.dropout = keras.layers.Dropout(config.dropout)
        self.classifier = keras.layers.Dense(2)

    def call(self, token_ids, training=False):
        x = self.embedding(token_ids)
        x = self.lstm(x, training=training)
        x = self.dropout(x, training=training)
        return self.classifier(x)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        if y is None or y_pred is None:
            return super().compute_loss(x, y, y_pred, sample_weight)
        return _compute_loss(y, y_pred, self.temperature)


class TransformerBlock(keras.layers.Layer):
    """
    Standard Transformer encoder block.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(d_ff, activation="relu"),
                keras.layers.Dense(d_model),
            ]
        )
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(ffn_out, training=training))


class SyscallTransformer(keras.Model):
    """
    Teacher transformer model implementation for the DongTing P1.5 pilot.
    """

    def __init__(self, config: TeacherConfig, window_size: int, **kwargs):
        super().__init__(**kwargs)
        self.temperature = config.temperature
        self.embedding = keras.layers.Embedding(config.vocab_size, config.d_model)

        # Position embedding
        self.pos_embedding = keras.layers.Embedding(window_size + 1, config.d_model)

        self.blocks = [
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        ]

        # We use sequence mean pooling since the data loader doesn't prepend a [CLS] token
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.classifier = keras.layers.Dense(2)

    def call(self, token_ids, training=False):
        seq_len = keras.ops.shape(token_ids)[1]
        positions = keras.ops.arange(start=0, stop=seq_len, step=1)

        x = self.embedding(token_ids) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x, training=training)

        pooled = self.pooling(x)
        return self.classifier(pooled)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        if y is None or y_pred is None:
            return super().compute_loss(x, y, y_pred, sample_weight)
        return _compute_loss(y, y_pred, self.temperature)
