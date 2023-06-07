import tensorflow as tf
import pandas as pd
import numpy as np


from typing import Any
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler


@dataclass
class NNModel:
    model: Any = None
    scalar: Any = None
    scalar_initialized: bool = False

    def __init__(self, input_dim: int = -1):
        assert input_dim > 0
        self.scalar = MinMaxScaler()
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", input_dim=input_dim),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        )

    def fit(self, X, y):
        if not self.scalar_initialized:
            X = self.scalar.fit_transform(X.copy())
            self.scalar_initialized = True
        else:
            X = X.copy()
        self.model.fit(
            x=X,
            y=y,
            verbose="auto",
            validation_split=0.1,
            shuffle=True,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2)],
        )

    def predict_proba(self, X):
        if self.scalar_initialized:
            X = self.scalar.transform(X.copy())
        y = self.model.predict(X)
        return y.flatten()

    def predict(self, X):
        return self.predict_proba(X) > 0.5
