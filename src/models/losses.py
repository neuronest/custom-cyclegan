from __future__ import annotations
from enum import Enum
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.losses import (
    binary_crossentropy,
    mean_squared_error,
    mean_absolute_error,
)


class GanLossCriterion(Enum):
    NEGATIVE_LOG_LIKELIHOOD = "binary_crossentropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"


def gan_loss(
    criterion: GanLossCriterion,
    discriminator_generation: tf.Tensor,
    discriminator_real: tf.Tensor,
    exclude_discriminator_term: bool,
) -> tf.Tensor:
    if criterion is GanLossCriterion.NEGATIVE_LOG_LIKELIHOOD:
        criterion_loss = binary_crossentropy
    elif criterion is GanLossCriterion.MEAN_SQUARED_ERROR:
        criterion_loss = mean_squared_error
    else:
        raise NotImplementedError
    generator_term = tf.reduce_mean(
        criterion_loss(
            discriminator_generation,
            tf.zeros(discriminator_generation.shape),
        )
    )
    if exclude_discriminator_term:
        return generator_term
    return generator_term + tf.reduce_mean(
        criterion_loss(
            discriminator_real,
            tf.ones(discriminator_real.shape),
        )
    )


def cycle_loss(
    image_A: tf.Tensor,
    image_B: tf.Tensor,
    image_A_reconstructed: tf.Tensor,
    image_B_reconstructed: tf.Tensor,
) -> tf.Tensor:
    return tf.reduce_mean(
        mean_absolute_error(image_A_reconstructed, image_A)
        + mean_absolute_error(image_B_reconstructed, image_B)
    )


def identity_loss(image: tf.Tensor, image_identity: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(mean_absolute_error(image_identity, image))


def get_generators_losses(
    image_A: tf.Tensor,
    image_B: tf.Tensor,
    forward_outputs: Tuple[tf.Tensor, ...],
    gan_loss_criterion: GanLossCriterion,
    cycle_loss_lambda: float,
    enable_identity_loss: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
    (
        image_A_reconstructed,
        discriminator_A_generation,
        discriminator_A_real,
        image_A_identity,
        image_B_reconstructed,
        discriminator_B_generation,
        discriminator_B_real,
        image_B_identity,
    ) = forward_outputs
    cycle_loss_all = cycle_loss(
        image_A=image_A,
        image_B=image_B,
        image_A_reconstructed=image_A_reconstructed,
        image_B_reconstructed=image_B_reconstructed,
    )
    gan_loss_A = gan_loss(
        criterion=gan_loss_criterion,
        discriminator_generation=discriminator_A_generation,
        discriminator_real=discriminator_A_real,
        exclude_discriminator_term=True,
    )
    gan_loss_B = gan_loss(
        criterion=gan_loss_criterion,
        discriminator_generation=discriminator_B_generation,
        discriminator_real=discriminator_B_real,
        exclude_discriminator_term=True,
    )
    generator_loss_A = gan_loss_A + tf.constant(cycle_loss_lambda) * cycle_loss_all
    generator_loss_B = gan_loss_B + tf.constant(cycle_loss_lambda) * cycle_loss_all
    if enable_identity_loss:
        generator_loss_A += identity_loss(
            image=image_A, image_identity=image_A_identity
        )
        generator_loss_B += identity_loss(
            image=image_B, image_identity=image_B_identity
        )
    return generator_loss_A, generator_loss_B


def get_discriminators_losses(
    forward_outputs: Tuple[tf.Tensor, ...],
    gan_loss_criterion: GanLossCriterion,
) -> Tuple[tf.Tensor, tf.Tensor]:
    (
        _,
        discriminator_A_generation,
        discriminator_A_real,
        _,
        _,
        discriminator_B_generation,
        discriminator_B_real,
        _,
    ) = forward_outputs
    discriminator_loss_A = gan_loss(
        criterion=gan_loss_criterion,
        discriminator_generation=discriminator_A_generation,
        discriminator_real=discriminator_A_real,
        exclude_discriminator_term=False,
    )
    discriminator_loss_B = gan_loss(
        criterion=gan_loss_criterion,
        discriminator_generation=discriminator_B_generation,
        discriminator_real=discriminator_B_real,
        exclude_discriminator_term=False,
    )
    return discriminator_loss_A, discriminator_loss_B


def get_losses(
    image_A: tf.Tensor,
    image_B: tf.Tensor,
    forward_outputs: Tuple[tf.Tensor, ...],
    gan_loss_criterion: GanLossCriterion,
    cycle_loss_lambda: float,
    enable_identity_loss: bool,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    generator_loss_A, generator_loss_B = get_generators_losses(
        image_A=image_A,
        image_B=image_B,
        forward_outputs=forward_outputs,
        gan_loss_criterion=gan_loss_criterion,
        cycle_loss_lambda=cycle_loss_lambda,
        enable_identity_loss=enable_identity_loss,
    )
    discriminator_loss_A, discriminator_loss_B = get_discriminators_losses(
        forward_outputs=forward_outputs,
        gan_loss_criterion=gan_loss_criterion,
    )
    return (
        generator_loss_A,
        generator_loss_B,
        discriminator_loss_A,
        discriminator_loss_B,
    )
