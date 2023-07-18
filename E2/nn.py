from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


@dataclass
class NNModel:
    model: Any = None
    scalar: Any = None
    scalar_initialized: bool = False
    num_epochs: int = 50
    patience: int = 5

    def __init__(self, input_dim: int = -1, num_epochs: int = 50, patience: int = 50):
        self.num_epochs = num_epochs
        self.patience = patience
        assert input_dim > 0
        tf.random.set_seed(0)
        np.random.seed(0)
        self.scalar = MinMaxScaler()
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, input_dim=input_dim),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(
                    32,
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(
                    16,
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(
                    8,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(
                    4,
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(
                    2,
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                ),
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
            validation_split=0.1,
            shuffle=True,
            batch_size=32,
            epochs=self.num_epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=self.patience)],
        )

    def predict_proba(self, X):
        if self.scalar_initialized:
            X = self.scalar.transform(X.copy())
        y = self.model.predict(X)
        return y.flatten()

    def predict(self, X):
        return self.predict_proba(X) > 0.5
