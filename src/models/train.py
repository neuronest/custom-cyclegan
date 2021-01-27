from box import Box
from typing import Tuple

import tensorflow as tf

from src.data_loader import ImageDataLoader, init_data_loaders
from src.models.architectures import CycleGAN
from src.models.losses import get_losses


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


def train(train_data_loader: ImageDataLoader, cfg: Box):
    cycle_gan = CycleGAN(
        **cfg.model,
        checkpoint_path=cfg.paths.checkpoint,
        max_checkpoints=cfg.training.max_checkpoints,
    )

    for epoch in range(cfg.training.epochs):
        for index_sample, (image_A, image_B) in enumerate(train_data_loader):
            image_A, image_B = tf.expand_dims(image_A, axis=0), tf.expand_dims(
                image_B, axis=0
            )
            (
                generator_loss_A,
                generator_loss_B,
                discriminator_loss_A,
                discriminator_loss_B,
            ) = train_step(cycle_gan, image_A, image_B)

            print(
                f"epoch: {epoch}\n",
                f"index sample: {index_sample}\n",
                f"generator_loss_A: {generator_loss_A.numpy().item()}\n",
                f"generator_loss_B: {generator_loss_B.numpy().item()}\n",
                f"discriminator_loss_A: {discriminator_loss_A.numpy().item()}\n",
                f"discriminator_loss_B: {discriminator_loss_B.numpy().item()}\n\n",
            )

        if epoch % cfg.training.frequency_checkpoints == 0:
            cycle_gan.save()

        train_data_loader, test_data_loader = init_data_loaders(cfg)
