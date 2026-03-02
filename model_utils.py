import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(num_classes):
    model = Sequential([
        # input_shape: (30 frames, 126 coordinates)
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model