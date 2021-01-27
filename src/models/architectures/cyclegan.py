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
        gan_loss_criterion: str,
        checkpoint_path: str,
        max_checkpoints: int,
        cycle_loss_lambda: float = 1e1,
        enable_identity_loss: bool = False,
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.gan_loss_criterion = GanLossCriterion(gan_loss_criterion)
        self.cycle_loss_lambda = cycle_loss_lambda
        self.enable_identity_loss = enable_identity_loss
        self.G = Generator(input_dim=input_dim, learning_rate=learning_rate)
        self.F = Generator(input_dim=input_dim, learning_rate=learning_rate)
        self.DA = Discriminator(input_dim=input_dim, learning_rate=learning_rate)
        self.DB = Discriminator(input_dim=input_dim, learning_rate=learning_rate)
        self.checkpoint, self.checkpoint_manager = self.init_checkpoint(
            checkpoint_path, max_checkpoints
        )

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

    @tf.function
    def forward(self, image_A: tf.Tensor, image_B: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        image_A_generated, image_B_generated = self.F(image_B), self.G(image_A)
        image_A_reconstructed, image_B_reconstructed = (
            self.G(image_A_generated),
            self.F(image_B_generated),
        )
        discriminator_A_real, discriminator_B_real = (
            self.DA(image_A),
            self.DB(image_B),
        )
        discriminator_A_generation, discriminator_B_generation = (
            self.DA(image_A_generated),
            self.DB(image_B_generated),
        )
        image_A_identity, image_B_identity = self.F(image_A), self.G(image_B)
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

    def save(self):
        self.checkpoint_manager.save()

    def load(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
