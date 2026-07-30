"""The proposal's BiLSTM classifier (3.5 / 6.6.1)."""


def build_bilstm(seq_len, n_features, n_classes):
    """Masking -> BiLSTM(64) -> BiLSTM(32) -> Dense(32) -> softmax. The Masking
    layer ignores zero-padded frames, so mask.npy need not be passed at fit time."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_features)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
