"""
models/transformer/mode_transformer.py

Режим C (задание научного руководителя):
  Трансформер, принимающий на вход моды (IMF от CEEMDAN или DWT-коэффициенты).

Архитектура:
  • Каждая мода стекируется как отдельный «канал» входного тензора
  • Positional Encoding + многоголовое Self-Attention (2 слоя)
  • Итоговый прогноз = линейный выход по всем каналам одновременно

Два источника мод:
  source='wavelet' → DWT (pywt)
  source='ceemdan' → CEEMDAN (PyEMD)

Требования: tensorflow>=2.13, pywt
"""
import warnings
import numpy as np
import pandas as pd
from typing import Literal, Optional

from analysis.metrics import calculate_metrics, infer_period


def _get_wavelet_modes(signal: np.ndarray, n_modes: int = 4, wavelet: str = "db4") -> np.ndarray:
    """
    DWT → первые n_modes восстановленных компонент.
    Returns: shape (n_modes, len(signal))
    """
    import pywt
    level = min(n_modes, pywt.dwt_max_level(len(signal), wavelet))
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    modes = []
    for i in range(min(n_modes, len(coeffs))):
        mask = [np.zeros_like(c) for c in coeffs]
        mask[i] = coeffs[i]
        rec = pywt.waverec(mask, wavelet)[:len(signal)]
        modes.append(rec)
    # Дополняем нулями если мод меньше n_modes
    while len(modes) < n_modes:
        modes.append(np.zeros(len(signal)))
    return np.array(modes)  # (n_modes, T)


def _get_ceemdan_modes(signal: np.ndarray, n_modes: int = 4) -> np.ndarray:
    """
    CEEMDAN → первые n_modes IMF компонент.
    Returns: shape (n_modes, len(signal))
    """
    from PyEMD import CEEMDAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cem = CEEMDAN(trials=30, noise_width=0.05)
        imfs = cem(signal.astype(float))
    # Берём первые n_modes
    selected = imfs[:n_modes]
    # Дополняем нулями если нужно
    pad = np.zeros((max(0, n_modes - len(selected)), len(signal)))
    return np.vstack([selected, pad]) if len(pad) > 0 else selected


def _build_transformer(
    seq_len: int,
    n_channels: int,
    forecast_horizon: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
):
    """
    Строит Keras-трансформер для многоканального временного ряда.

    Вход: (batch, seq_len, n_channels)
    Выход: (batch, forecast_horizon)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(seq_len, n_channels))

    # Проекция каналов в d_model
    x = layers.Dense(d_model)(inputs)

    # Positional Encoding (синусоидальное)
    positions = tf.range(seq_len, dtype=tf.float32)
    pos_enc = _positional_encoding(seq_len, d_model)
    x = x + pos_enc

    # Transformer блоки
    for _ in range(n_layers):
        # Multi-Head Self-Attention
        attn_out = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout,
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        # Feed-Forward
        ff = layers.Dense(d_model * 2, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    # Глобальное усреднение по времени
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d_model // 2, activation="relu")(x)
    outputs = layers.Dense(forecast_horizon)(x)

    return Model(inputs=inputs, outputs=outputs)


def _positional_encoding(seq_len: int, d_model: int):
    """Синусоидальное positional encoding."""
    import tensorflow as tf
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def _prepare_sequences(
    modes: np.ndarray,          # (n_modes, T)
    seq_len: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Создаёт обучающие пары (X, y) из многоканального ряда.
    X: (samples, seq_len, n_modes)
    y: (samples, forecast_horizon) — сумма всех мод (=исходный ряд)
    """
    T = modes.shape[1]
    n_channels = modes.shape[0]
    original = modes.sum(axis=0)  # восстановленный ряд

    X_list, y_list = [], []
    for i in range(T - seq_len - forecast_horizon + 1):
        X_list.append(modes[:, i : i + seq_len].T)  # (seq_len, n_channels)
        y_list.append(original[i + seq_len : i + seq_len + forecast_horizon])

    return np.array(X_list), np.array(y_list)


def mode_transformer_forecast(
    series: pd.Series,
    test_size: int = 18,
    source: Literal["wavelet", "ceemdan"] = "wavelet",
    n_modes: int = 4,
    wavelet: str = "db4",
    seq_len: Optional[int] = None,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 16,
) -> tuple[pd.Series, dict]:
    """
    Режим C: трансформер на модах DWT или CEEMDAN.

    Parameters
    ----------
    series : pd.Series
    test_size : горизонт прогноза
    source : 'wavelet' (DWT) | 'ceemdan' (CEEMDAN)
    n_modes : сколько мод подать на вход трансформеру
    wavelet : вейвлет для DWT (при source='wavelet')
    seq_len : длина контекстного окна (None = 2*m)
    d_model : размерность модели трансформера
    n_heads : число голов внимания
    n_layers : число блоков трансформера
    epochs : макс. число эпох
    batch_size : размер батча

    Returns
    -------
    (forecast_series, metrics_dict)
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler

    train_vals = series.values[:-test_size].astype(float)
    test_vals  = series.values[-test_size:].astype(float)
    m = infer_period(series)

    if seq_len is None:
        seq_len = max(2 * m, 24)

    min_required = seq_len + test_size + 10
    if len(train_vals) < min_required:
        raise ValueError(
            f"Ряд слишком короткий для трансформера: {len(train_vals)} < {min_required}"
        )

    # Нормализация
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()

    # Получаем моды
    if source == "ceemdan":
        modes = _get_ceemdan_modes(train_scaled, n_modes=n_modes)
        model_name = f"Transformer(CEEMDAN,{n_modes})"
    else:
        modes = _get_wavelet_modes(train_scaled, n_modes=n_modes, wavelet=wavelet)
        model_name = f"Transformer(Wavelet,{n_modes})"

    # Подготовка обучающих данных
    X, y = _prepare_sequences(modes, seq_len=seq_len, forecast_horizon=test_size)
    if len(X) < 10:
        raise ValueError(f"Недостаточно обучающих примеров: {len(X)}")

    # Нормализация таргетов
    y_scaler = StandardScaler()
    y_flat = y.flatten()
    y_norm = y_scaler.fit_transform(y_flat.reshape(-1, 1)).flatten().reshape(y.shape)

    # Строим и обучаем трансформер
    tf.random.set_seed(42)
    model = _build_transformer(
        seq_len=seq_len,
        n_channels=n_modes,
        forecast_horizon=test_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model.compile(optimizer="adam", loss="mse")

    cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X, y_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[cb],
            verbose=0,
        )

    # Прогноз: последнее окно из мод
    last_window = modes[:, -seq_len:].T[np.newaxis]  # (1, seq_len, n_modes)
    pred_norm = model.predict(last_window, verbose=0)[0]
    forecast = y_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

    metrics = calculate_metrics(test_vals, forecast, train_vals, m)
    metrics["Model"] = model_name

    return pd.Series(forecast, index=series.index[-test_size:]), metrics
