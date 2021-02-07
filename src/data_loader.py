import inspect
import os
import sys
from abc import ABC, abstractmethod
from itertools import cycle
from functools import partial
from typing import Sequence, Tuple, Union, Optional

import numpy as np
import tensorflow as tf
from box import Box
from tensorflow.image import resize, random_crop
from tensorflow.keras.preprocessing.image import load_img


class DataPaths:
    def __init__(
        self, root_path_A: str, root_path_B: str, extension: str, shuffle: bool
    ):
        self.root_path_A, self.root_path_B = (
            root_path_A,
            root_path_B,
        )
        self.extension = extension
        self.shuffle = shuffle
        self.paths_A = [
            filename
            for filename in os.listdir(self.root_path_A)
            if filename.endswith(extension)
        ]
        self.paths_B = [
            filename
            for filename in os.listdir(self.root_path_B)
            if filename.endswith(extension)
        ]
        if self.shuffle:
            np.random.shuffle(self.paths_A)
            np.random.shuffle(self.paths_B)
        self.length = min(len(self.paths_A), len(self.paths_B))

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> "DataPaths":
        self.paths_A, self.paths_B = cycle(self.paths_A), cycle(self.paths_B)
        return self

    def __next__(self) -> Tuple[str, str]:
        current_path_A, current_path_B = next(self.paths_A), next(self.paths_B)
        return os.path.join(self.root_path_A, current_path_A), os.path.join(
            self.root_path_B, current_path_B
        )


class ImageDataLoader(ABC):
    def __init__(
        self,
        data_paths: DataPaths,
        target_size: Tuple[int, ...],
        crop: bool,
        enable_normalization: bool,
        before_crop_size: Optional[Tuple[int, ...]],
        interpolation: str,
    ):
        self.data_paths = data_paths
        self.target_size = target_size
        self.crop = crop
        self.enable_normalization = enable_normalization
        self.before_crop_size = before_crop_size
        self.interpolation = interpolation
        if self.crop:
            self.inner_loader = load_img
        else:
            self.inner_loader = partial(
                load_img, target_size=target_size, interpolation=interpolation
            )

    def __len__(self) -> int:
        return len(self.data_paths)

    @abstractmethod
    def __iter__(self) -> "ImageDataLoader":
        raise NotImplementedError

    @abstractmethod
    def load(self) -> "ImageDataLoader":
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        return (image / 127.5) - 1

    @staticmethod
    def denormalize(image: np.ndarray) -> np.ndarray:
        return (image + 1) * 127.5

    def next_batch(
        self, n: int, as_tensors: bool = False
    ) -> Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]]:
        batch_A, batch_B = np.array([next(self) for _ in range(n)]).swapaxes(0, 1)
        if as_tensors:
            batch_A, batch_B = tf.convert_to_tensor(batch_A), tf.convert_to_tensor(
                batch_B
            )
        return batch_A, batch_B

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.crop:
            assert self.before_crop_size is not None
            resize(image, self.before_crop_size, method=self.interpolation)
            random_crop(image, size=self.target_size)
        if self.enable_normalization:
            image = (image / 127.5) - 1
        return image.astype("float32")


class MemoryDataLoader(ImageDataLoader):
    name = "MEMORY"

    def __init__(
        self,
        data_paths: DataPaths,
        target_size: Tuple[int, ...],
        crop: bool,
        enable_normalization: bool,
        before_crop_size: Optional[Tuple[int, ...]],
        interpolation: str,
    ):
        super().__init__(
            data_paths,
            target_size,
            crop,
            enable_normalization,
            before_crop_size,
            interpolation,
        )
        self.images_A, self.images_B = None, None

    def __iter__(self) -> ImageDataLoader:
        try:
            self.images_A, self.images_B = iter(self.images_A), iter(self.images_B)
        except TypeError:
            pass
        return self

    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return next(self.images_A), next(self.images_B)

    def load(self) -> ImageDataLoader:
        self.images_A, self.images_B = [], []
        for current_path_A, current_path_B in self.data_paths:
            self.images_A.append(
                self.preprocess(np.array(self.inner_loader(current_path_A)))
            )
            self.images_B.append(
                self.preprocess(np.array(self.inner_loader(current_path_B)))
            )
        self.images_A = tf.convert_to_tensor(self.images_A)
        self.images_B = tf.convert_to_tensor(self.images_B)
        return self


class GeneratorDataLoader(ImageDataLoader):
    name = "GENERATOR"

    def __init__(
        self,
        data_paths: DataPaths,
        target_size: Tuple[int, ...],
        crop: bool,
        enable_normalization: bool,
        before_crop_size: Tuple[int, ...],
        interpolation: str,
    ):
        super().__init__(
            data_paths,
            target_size,
            crop,
            enable_normalization,
            before_crop_size,
            interpolation,
        )

    def __iter__(self) -> ImageDataLoader:
        return self

    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor]:
        current_path_A, current_path_B = next(self.data_paths)
        return (
            tf.convert_to_tensor(
                self.preprocess(np.array(self.inner_loader(current_path_A)))
            ),
            tf.convert_to_tensor(
                self.preprocess(np.array(self.inner_loader(current_path_B)))
            ),
        )

    def load(self) -> ImageDataLoader:
        self.data_paths = iter(self.data_paths)
        return self


class DataLoaderFactory:
    _data_loaders = {
        cls.name: cls
        for _, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if ImageDataLoader in cls.__bases__
    }

    @classmethod
    def new(
        cls,
        name: str,
        data_paths: DataPaths,
        target_size: Sequence[int],
        crop: bool,
        enable_normalization: bool,
        before_crop_size: Optional[Sequence[int]] = None,
        interpolation: str = "bicubic",
    ) -> ImageDataLoader:
        return cls._data_loaders[name](
            data_paths=data_paths,
            target_size=tuple(target_size),
            crop=crop,
            enable_normalization=enable_normalization,
            before_crop_size=tuple(before_crop_size) if before_crop_size else None,
            interpolation=interpolation,
        )


def init_data_loaders(cfg: Box) -> Tuple[ImageDataLoader, ImageDataLoader]:
    train_data_paths = DataPaths(
        root_path_A=cfg.paths.data.train_A,
        root_path_B=cfg.paths.data.train_B,
        **cfg.data_paths,
    )
    train_data_loader = DataLoaderFactory.new(
        **cfg.data_loader, data_paths=train_data_paths
    ).load()

    test_data_paths_config = cfg.data_paths.to_dict()
    test_data_paths_config["shuffle"] = False
    test_data_paths = DataPaths(
        root_path_A=cfg.paths.data.test_A,
        root_path_B=cfg.paths.data.test_B,
        **test_data_paths_config,
    )
    test_data_loader_config = cfg.data_loader.to_dict()
    test_data_loader_config["crop"] = False
    test_data_loader_config["before_crop_size"] = None
    test_data_loader = DataLoaderFactory.new(
        **test_data_loader_config, data_paths=test_data_paths
    ).load()
    return train_data_loader, test_data_loader
