from typing import Optional, Callable, Union

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    UpSampling2D,
    Activation,
    ZeroPadding2D,
    add,
)
from tensorflow_addons.layers import InstanceNormalization


class ReflectionPadding2D(Layer):
    def __init__(self, padding: int, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, x, mask=None):
        # no padding for batch and channel dimensions
        return tf.pad(
            x,
            [
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            "REFLECT",
        )


class ConvolutionBlock(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        activation: Optional[Union[Callable, str]] = None,
        padding_layer: Optional[Callable] = None,
        padding: Optional[int] = None,
        instance_normalization: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if padding_layer is not None and padding is not None:
            self.reflexion_padding_layer = padding_layer(padding=padding)
        else:
            self.reflexion_padding_layer = None
        self.convolution_layer = Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides
        )
        if instance_normalization:
            self.instance_normalization_layer = InstanceNormalization(axis=3)
        else:
            self.instance_normalization_layer = None
        if callable(activation):
            self.activation_layer = activation
        else:
            self.activation_layer = Activation(activation)

    def call(self, inputs, training=False):
        outputs = inputs
        if self.reflexion_padding_layer is not None:
            outputs = self.reflexion_padding_layer(outputs)
        outputs = self.convolution_layer(outputs)
        if self.instance_normalization_layer is not None:
            outputs = self.instance_normalization_layer(outputs, training=training)
        outputs = self.activation_layer(outputs)
        return outputs


class DownsamplingBlock(ConvolutionBlock):
    def __init__(self, filters: int, **kwargs):
        super().__init__(
            filters,
            kernel_size=3,
            strides=2,
            activation="relu",
            padding_layer=ZeroPadding2D,
            padding=1,
            instance_normalization=True,
            **kwargs
        )


class UpsamplingBlock(ConvolutionBlock):
    def __init__(self, filters: int, **kwargs):
        # https://distill.pub/2016/deconv-checkerboard/
        super().__init__(
            filters,
            kernel_size=3,
            strides=1,
            activation="relu",
            padding_layer=ReflectionPadding2D,
            padding=1,
            instance_normalization=True,
            **kwargs
        )
        self.upsampling_layer = UpSampling2D(size=(2, 2))

    def call(self, inputs, training=False):
        return super().call(self.upsampling_layer(inputs))


class ResidualBlock(Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.first_convolution_block = ConvolutionBlock(
            filters,
            kernel_size=3,
            activation="relu",
            padding_layer=ReflectionPadding2D,
            padding=1,
            instance_normalization=True,
        )
        self.second_convolution_block = ConvolutionBlock(
            filters,
            kernel_size=3,
            activation=None,
            padding_layer=ReflectionPadding2D,
            padding=1,
            instance_normalization=True,
        )

    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.first_convolution_block(outputs, training=training)
        outputs = self.second_convolution_block(outputs, training=training)
        return add([inputs, outputs])
