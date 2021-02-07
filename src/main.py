import logging

from src.config import cfg
from src.data_loader import init_data_loaders
from src.models.train import train

if __name__ == "__main__":
    logging.basicConfig(
        format=cfg.logging.format,
        level=cfg.logging.level,
    )
    train_data_loader, test_data_loader = init_data_loaders(cfg)
    train(train_data_loader, cfg)
