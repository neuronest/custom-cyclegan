import random
from collections import deque
from typing import Tuple

import tensorflow as tf

from src.models.architectures.generator import Generator
from src.models.architectures.discriminator import Discriminator
from src.models.losses import GanLossCriterion


class CycleGAN:
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        learning_rate: float,
        adam_beta_1: float,
        gan_loss_criterion: str,
        checkpoint_path: str,
        max_checkpoints: int,
        generated_images_buffer_size: int,
        cycle_loss_lambda: float = 1e1,
        enable_identity_loss: bool = False,
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.gan_loss_criterion = GanLossCriterion(gan_loss_criterion)
        self.cycle_loss_lambda = cycle_loss_lambda
        self.enable_identity_loss = enable_identity_loss
        networks_parameters = dict(
            input_dim=self.input_dim,
            learning_rate=self.learning_rate,
            adam_beta_1=self.adam_beta_1,
        )
        self.G = Generator(**networks_parameters)
        self.F = Generator(**networks_parameters)
        self.DA = Discriminator(**networks_parameters)
        self.DB = Discriminator(**networks_parameters)
        self.checkpoint, self.checkpoint_manager = self.init_checkpoint(
            checkpoint_path, max_checkpoints
        )
        self.generated_images_buffer = deque([], maxlen=generated_images_buffer_size)

    def init_checkpoint(
        self, checkpoint_path: str, max_checkpoints: int
    ) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:
        checkpoint = tf.train.Checkpoint(
            G=self.G.model,
            F=self.F.model,
            DA=self.DA.model,
            DB=self.DB.model,
            G_optimizer=self.G.optimizer,
            F_optimizer=self.F.optimizer,
            DA_optimizer=self.DA.optimizer,
            DB_optimizer=self.DB.optimizer,
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_path, max_to_keep=max_checkpoints
        )
        return checkpoint, checkpoint_manager

    def forward(self, image_A: tf.Tensor, image_B: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        image_A_generated, image_B_generated = self.F(image_B), self.G(image_A)
        image_A_identity, image_B_identity = self.F(image_A), self.G(image_B)
        image_A_reconstructed, image_B_reconstructed = (
            self.F(image_B_generated),
            self.G(image_A_generated),
        )
        self.generated_images_buffer.append(
            (image_A_generated, image_B_generated, image_A_identity, image_B_identity)
        )
        (
            image_A_generated,
            image_B_generated,
            image_A_identity,
            image_B_identity,
        ) = random.choice(self.generated_images_buffer)
        discriminator_A_real, discriminator_B_real = (
            self.DA(image_A),
            self.DB(image_B),
        )
        discriminator_A_generation, discriminator_B_generation = (
            self.DA(image_A_generated),
            self.DB(image_B_generated),
        )
        return (
            image_A_reconstructed,
            discriminator_A_generation,
            discriminator_A_real,
            image_A_identity,
            image_B_reconstructed,
            discriminator_B_generation,
            discriminator_B_real,
            image_B_identity,
        )

    def _get_gradients(
        self,
        losses: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        tape: tf.GradientTape,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        (
            generator_loss_A,
            generator_loss_B,
            discriminator_loss_A,
            discriminator_loss_B,
        ) = losses
        G_gradients = tape.gradient(generator_loss_A, self.G.model.trainable_variables)
        F_gradients = tape.gradient(generator_loss_B, self.F.model.trainable_variables)
        DA_gradients = tape.gradient(
            discriminator_loss_A, self.DA.model.trainable_variables
        )
        DB_gradients = tape.gradient(
            discriminator_loss_B, self.DB.model.trainable_variables
        )
        return (
            G_gradients,
            F_gradients,
            DA_gradients,
            DB_gradients,
        )

    def _apply_gradients(
        self, gradients: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ):
        G_gradients, F_gradients, DA_gradients, DB_gradients = gradients
        self.G.optimizer.apply_gradients(
            zip(G_gradients, self.G.model.trainable_variables)
        )
        self.F.optimizer.apply_gradients(
            zip(F_gradients, self.F.model.trainable_variables)
        )
        self.DA.optimizer.apply_gradients(
            zip(DA_gradients, self.DA.model.trainable_variables)
        )
        self.DB.optimizer.apply_gradients(
            zip(DB_gradients, self.DB.model.trainable_variables)
        )

    def backward(
        self,
        losses: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        tape: tf.GradientTape,
    ):
        gradients = self._get_gradients(losses=losses, tape=tape)
        self._apply_gradients(gradients=gradients)

    def save(self):
        self.checkpoint_manager.save()

    def load(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
