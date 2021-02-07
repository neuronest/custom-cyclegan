import logging
import os
from box import Box
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import tensorflow as tf

from src.utils import render_images
from src.data_loader import ImageDataLoader
from src.models.architectures import CycleGAN
from src.models.losses import get_losses

logger = logging.getLogger(__name__)


@dataclass
class AverageLoss:
    incremental_generator_A: float = 0
    incremental_generator_B: float = 0
    incremental_discriminator_A: float = 0
    incremental_discriminator_B: float = 0
    n: float = 0

    def append(
        self,
        generator_loss_A: tf.Tensor,
        generator_loss_B: tf.Tensor,
        discriminator_loss_A: tf.Tensor,
        discriminator_loss_B: tf.Tensor,
    ):
        self.n += 1
        self.incremental_generator_A += (
            generator_loss_A.numpy().item() - self.incremental_generator_A
        ) / self.n
        self.incremental_generator_B += (
            generator_loss_B.numpy().item() - self.incremental_generator_B
        ) / self.n
        self.incremental_discriminator_A += (
            discriminator_loss_A.numpy().item() - self.incremental_discriminator_A
        ) / self.n
        self.incremental_discriminator_B += (
            discriminator_loss_B.numpy().item() - self.incremental_discriminator_B
        ) / self.n


@tf.function
def train_step(
    cycle_gan: CycleGAN, image_A: tf.Tensor, image_B: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape(persistent=True) as tape:
        forward_outputs = cycle_gan.forward(image_A, image_B)
        losses = get_losses(
            image_A=image_A,
            image_B=image_B,
            forward_outputs=forward_outputs,
            gan_loss_criterion=cycle_gan.gan_loss_criterion,
            cycle_loss_lambda=cycle_gan.cycle_loss_lambda,
            enable_identity_loss=cycle_gan.enable_identity_loss,
        )
    cycle_gan.backward(losses, tape)
    return losses


def inner_train_loop(
    cycle_gan: CycleGAN, train_data_loader: ImageDataLoader, epoch: int
) -> AverageLoss:
    average_loss = AverageLoss()
    for index_sample in range(len(train_data_loader)):
        image_A, image_B = next(train_data_loader)
        image_A, image_B = tf.expand_dims(image_A, axis=0), tf.expand_dims(
            image_B, axis=0
        )
        (
            generator_loss_A,
            generator_loss_B,
            discriminator_loss_A,
            discriminator_loss_B,
        ) = train_step(cycle_gan, image_A, image_B)
        average_loss.append(
            generator_loss_A,
            generator_loss_B,
            discriminator_loss_A,
            discriminator_loss_B,
        )
        logger.info(
            "epoch: %d, index_sample: %d, "
            "generator_loss_A: %.4f, generator_loss_B: %.4f, "
            "discriminator_loss_A: %.4f, discriminator_loss_B: %.4f",
            epoch,
            index_sample,
            average_loss.incremental_generator_A,
            average_loss.incremental_generator_B,
            average_loss.incremental_discriminator_A,
            average_loss.incremental_discriminator_B,
        )
    return average_loss


def train(train_data_loader: ImageDataLoader, cfg: Box):
    tensorboard_path = os.path.join(
        cfg.paths.tensorboard, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_path)
    cycle_gan = CycleGAN(
        **cfg.model,
        checkpoint_path=cfg.paths.checkpoint,
        max_checkpoints=cfg.training.max_checkpoints,
    )

    for epoch in range(cfg.training.epochs):
        average_loss = inner_train_loop(cycle_gan, train_data_loader, epoch)
        if epoch % cfg.training.frequency_checkpoints == 0:
            cycle_gan.save()
            example_images_A, example_images_B = train_data_loader.next_batch(
                cfg.training.example_images_number_checkpoint, as_tensors=True
            )
            example_images_A_generated, example_images_B_generated = (
                cycle_gan.F(example_images_B),
                cycle_gan.G(example_images_A),
            )
            with tensorboard_writer.as_default():
                tf.summary.scalar(
                    "generator loss A", average_loss.incremental_generator_A, step=epoch
                )
                tf.summary.scalar(
                    "generator loss B", average_loss.incremental_generator_B, step=epoch
                )
                tf.summary.scalar(
                    "discriminator loss A",
                    average_loss.incremental_discriminator_B,
                    step=epoch,
                )
                tf.summary.scalar(
                    "discriminator loss B",
                    average_loss.incremental_discriminator_A,
                    step=epoch,
                )
                tf.summary.image(
                    "training A to B",
                    render_images(example_images_A, example_images_B_generated),
                    step=epoch,
                )
                tf.summary.image(
                    "training B to A",
                    render_images(example_images_B, example_images_A_generated),
                    step=epoch,
                )
