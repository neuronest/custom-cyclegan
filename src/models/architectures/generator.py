from typing import Tuple

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.models.architectures.base import (
    ReflectionPadding2D,
    ConvolutionBlock,
    DownsamplingBlock,
    UpsamplingBlock,
    ResidualBlock,
)


class Generator:
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        learning_rate: float,
        adam_beta_1: float,
        minimum_filters: int = 64,
        residual_filter: int = 256,
        residual_number: int = 9,
        final_filters: int = 3,
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.minimum_filters = minimum_filters
        self.residual_filter = residual_filter
        self.residual_number = residual_number
        self.final_filters = final_filters
        self.optimizer = Adam(self.learning_rate, beta_1=self.adam_beta_1)
        self.initial_layer = ConvolutionBlock(
            filters=minimum_filters,
            kernel_size=7,
            activation="relu",
            padding_layer=ReflectionPadding2D,
            padding=3,
            instance_normalization=True,
        )
        self.downsampling_layers = [
            DownsamplingBlock(filters)
            for filters in [
                2 ** i
                for i in range(
                    np.log2(minimum_filters).astype(int) + 1,
                    np.log2(residual_filter).astype(int) + 1,
                )
            ]
        ]
        self.residual_layers = [
            ResidualBlock(residual_filter) for _ in range(residual_number)
        ]
        self.upsampling_layers = [
            UpsamplingBlock(filters)
            for filters in [
                2 ** i
                for i in range(
                    np.log2(residual_filter).astype(int) - 1,
                    np.log2(minimum_filters).astype(int) - 1,
                    -1,
                )
            ]
        ]
        self.final_layer = ConvolutionBlock(
            filters=final_filters,
            kernel_size=7,
            activation="tanh",
            padding_layer=ReflectionPadding2D,
            padding=3,
            instance_normalization=False,
        )
        self.model = self.build()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def build(self) -> Model:
        inputs = Input(shape=self.input_dim)
        outputs = inputs
        outputs = self.initial_layer(outputs)
        for downsampling_layer in self.downsampling_layers:
            outputs = downsampling_layer(outputs)
        for residual_layer in self.residual_layers:
            outputs = residual_layer(outputs)
        for upsampling_layer in self.upsampling_layers:
            outputs = upsampling_layer(outputs)
        outputs = self.final_layer(outputs)
        model = Model(inputs, outputs)
        return model
