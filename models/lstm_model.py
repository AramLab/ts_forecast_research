"""
models/lstm_model.py
LSTM прогнозирование через TensorFlow/Keras.

Два режима:
  fast=False (default) - полная модель 2×LSTM(50), ~20-40с на ряд
  fast=True            - лёгкая модель 1×LSTM(16), ~3-8с, для CEEMDAN
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from analysis.metrics import calculate_metrics, infer_period


def lstm_forecast(
    series: pd.Series,
    test_size: int = 18,
    seq_length: int = None,
    epochs: int = 50,
    batch_size: int = 16,
    units: int = 50,
    fast: bool = False,
) -> tuple[pd.Series, dict]:
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError("TensorFlow не установлен.")

    if fast:
        units = 16
        epochs = 30

    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]
    m = infer_period(series)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    if seq_length is None:
        seq_length = min(m, len(scaled) // 4)
    seq_length = max(seq_length, 2)

    if len(scaled) <= seq_length + 5:
        raise ValueError(f"Ряд слишком короткий для LSTM: {len(scaled)}")

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length: i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)

    if fast:
        # Лёгкая архитектура
        model = Sequential([
            LSTM(units, input_shape=(seq_length, 1)),
            Dense(1),
        ])
    else:
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(units, return_sequences=False),
            Dropout(0.2),
            Dense(max(units // 2, 8)),
            Dense(1),
        ])

    model.compile(optimizer="adam", loss="mse")
    cb = EarlyStopping(monitor="val_loss", patience=5,
                       restore_best_weights=True, verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, callbacks=[cb], verbose=0)

    last_seq = scaled[-seq_length:].reshape(1, seq_length, 1)
    preds = []
    for _ in range(test_size):
        p = model.predict(last_seq, verbose=0)[0, 0]
        preds.append(p)
        last_seq = np.concatenate(
            [last_seq[:, 1:, :], np.array([[[p]]])], axis=1)

    forecast = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)).flatten()

    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = "LSTM"
    return pd.Series(forecast, index=test.index), metrics


def lstm_fast(series: pd.Series, test_size: int = 18):
    return lstm_forecast(series, test_size=test_size, fast=True)
