"""Загрузка, разбиение и стандартизация данных"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Подготовленные обучающая и тестовая выборки"""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_and_prepare(
    data_path: Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> DatasetSplit:
    """Загружает Raisin и подготавливает данные для эксперимента"""
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    if data.shape != (900, 8):
        raise ValueError("Ожидалась таблица размером 900 на 8")
    if not np.isfinite(data).all():
        raise ValueError("Данные содержат пропуски или бесконечные значения")

    X = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("Целевая переменная должна содержать классы 0 и 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Статистики считаются только по обучающей выборке, чтобы избежать утечки
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)

    return DatasetSplit(
        X_train=(X_train - mean) / scale,
        X_test=(X_test - mean) / scale,
        y_train=y_train,
        y_test=y_test,
    )
