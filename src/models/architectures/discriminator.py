from typing import Tuple

import numpy as np
from tensorflow.keras.layers import Input, LeakyReLU, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.models.architectures.base import ConvolutionBlock


class Discriminator:
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        learning_rate: float,
        adam_beta_1: float,
        minimum_filters: int = 64,
        maximal_filters: int = 512,
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.optimizer = Adam(self.learning_rate, beta_1=self.adam_beta_1)
        self.convolutional_layers = [
            ConvolutionBlock(
                filters=filters,
                kernel_size=4,
                strides=1 if filters == maximal_filters else 2,
                activation=LeakyReLU(alpha=0.2),
                padding_layer=ZeroPadding2D,
                padding=1,
                instance_normalization=index > 0,
            )
            for index, filters in enumerate(
                [
                    2 ** i
                    for i in range(
                        np.log2(minimum_filters).astype(int),
                        np.log2(maximal_filters).astype(int) + 1,
                    )
                ]
            )
        ]
        self.final_layer = ConvolutionBlock(
            filters=1,
            kernel_size=4,
            padding_layer=ZeroPadding2D,
            padding=1,
            instance_normalization=False,
        )
        self.model = self.build()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def build(self) -> Model:
        inputs = Input(shape=self.input_dim)
        outputs = inputs
        for convolutional_layer in self.convolutional_layers:
            outputs = convolutional_layer(outputs)
        outputs = self.final_layer(outputs)
        model = Model(inputs, outputs)
        return model
