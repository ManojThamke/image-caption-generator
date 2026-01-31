# src/model.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Add, Dropout, Activation, RepeatVector
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import (
    VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, MAX_CAPTION_LENGTH
)

# ==============================
# SIMPLE ATTENTION MECHANISM
# ==============================
def attention_block(image_features, lstm_output):
    """
    image_features: (batch, 2048)
    lstm_output: (batch, lstm_units)
    """
    image_dense = Dense(LSTM_UNITS, activation="relu")(image_features)
    merged = Add()([image_dense, lstm_output])
    attention = Dense(LSTM_UNITS, activation="tanh")(merged)
    return attention


# ==============================
# BUILD MODEL
# ==============================
def build_model():
    # -------- Image features input --------
    image_input = Input(shape=(2048,), name="image_features")
    img_dense = Dense(EMBEDDING_DIM, activation="relu")(image_input)
    img_dense = Dropout(0.5)(img_dense)

    # -------- Text input --------
    text_input = Input(shape=(MAX_CAPTION_LENGTH,), name="caption_input")
    embedding = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(text_input)

    lstm_out = LSTM(LSTM_UNITS)(embedding)

    # -------- Attention --------
    attn_out = attention_block(img_dense, lstm_out)

    # -------- Combine & output --------
    combined = Add()([img_dense, attn_out])
    combined = Dense(LSTM_UNITS, activation="relu")(combined)
    outputs = Dense(VOCAB_SIZE, activation="softmax")(combined)

    # -------- Compile model --------
    model = Model(inputs=[image_input, text_input], outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001)
    )

    return model


# ==============================
# MAIN (TEST BUILD)
# ==============================
if __name__ == "__main__":
    model = build_model()
    model.summary()
