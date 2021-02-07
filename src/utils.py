import numpy as np
import tensorflow as tf

from src.data_loader import ImageDataLoader


def render_images(
    images_real: tf.Tensor,
    images_predicted: tf.Tensor,
    denormalize: bool = True,
    h_border_size: int = 10,
    v_border_size: int = 10,
) -> np.ndarray:
    def _hstack_images_with_border(
        images: np.ndarray, border: np.ndarray
    ) -> np.ndarray:
        return np.hstack(
            [elem for image in images[:-1] for elem in (image, border)] + [images[-1]]
        )

    image_height = images_real.shape[1]
    images_real, images_predicted = images_real.numpy(), images_predicted.numpy()
    if denormalize:
        images_real, images_predicted = (
            ImageDataLoader.denormalize(images_real),
            ImageDataLoader.denormalize(images_predicted),
        )
    images_real, images_predicted = tuple(images_real), tuple(images_predicted)
    h_border = np.ones((image_height, h_border_size, 3)) * 255
    images_real = _hstack_images_with_border(images_real, h_border)
    images_predicted = _hstack_images_with_border(images_predicted, h_border)
    v_border = np.ones((v_border_size, images_real.shape[1], 3)) * 255
    return np.expand_dims(
        np.vstack([images_real, v_border, images_predicted]).astype("uint8"), axis=0
    )
