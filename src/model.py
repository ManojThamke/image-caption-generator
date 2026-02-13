# src/model.py

from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import (
    VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, MAX_CAPTION_LENGTH
)

# ==============================
# BUILD IMAGE CAPTIONING MODEL
# ==============================
def build_model():
    # -------- Image feature input --------
    image_input = Input(shape=(2048,), name="image_features")
    image_dense = Dense(EMBEDDING_DIM, activation="relu")(image_input)
    image_dense = Dropout(0.5)(image_dense)

    # -------- Text input --------
    caption_input = Input(
        shape=(MAX_CAPTION_LENGTH,),
        name="caption_input"
    )

    caption_embedding = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(caption_input)

    # -------- LSTM Decoder --------
    lstm_output = LSTM(
        LSTM_UNITS,
        dropout=0.5,
        recurrent_dropout=0.3
    )(caption_embedding, initial_state=[image_dense, image_dense])

    # -------- Output layer --------
    outputs = Dense(VOCAB_SIZE, activation="softmax")(lstm_output)

    # -------- Compile model --------
    model = Model(
        inputs=[image_input, caption_input],
        outputs=outputs
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005)
    )

    return model


# ==============================
# MAIN (TEST BUILD)
# ==============================
if __name__ == "__main__":
    model = build_model()
    model.summary()
