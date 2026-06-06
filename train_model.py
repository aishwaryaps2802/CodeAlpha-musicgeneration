# train_model.py
# Build and train the LSTM model

import numpy as np
import pickle
from tensorflow.keras.models        import Sequential
from tensorflow.keras.layers        import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks     import ModelCheckpoint, EarlyStopping
from prepare_data                   import get_notes, prepare_sequences

def build_model(n_vocab, sequence_length=100):
    model = Sequential([
        # First LSTM layer
        LSTM(
            256,
            input_shape = (sequence_length, 1),
            return_sequences = True
        ),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer
        LSTM(256),
        BatchNormalization(),
        Dropout(0.3),

        # Dense output layers
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])

    model.compile(
        loss      = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics   = ['accuracy']
    )

    print(model.summary())
    return model

def train(epochs=50, batch_size=64):
    # Load and prepare data
    notes               = get_notes()
    X, y, n_vocab, _, _ = prepare_sequences(notes)

    # Build model
    model = build_model(n_vocab)

    # Save best model automatically
    checkpoint = ModelCheckpoint(
        'model/music_model.h5',
        monitor           = 'loss',
        verbose           = 1,
        save_best_only    = True,
        mode              = 'min'
    )

    # Stop early if no improvement
    early_stop = EarlyStopping(
        monitor   = 'loss',
        patience  = 10,
        verbose   = 1
    )

    print("\nStarting training...")
    model.fit(
        X, y,
        epochs     = epochs,
        batch_size = batch_size,
        callbacks  = [checkpoint, early_stop]
    )

    print("\nTraining complete! Model saved to model/music_model.h5")

if __name__ == '__main__':
    import os
    os.makedirs('model', exist_ok=True)
    train()