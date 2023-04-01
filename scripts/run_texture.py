import pyrallis

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure
import os
from loguru import logger
import torch
from src import utils

@pyrallis.wrap()
def main(cfg: TrainConfig):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
    



    trainer = TEXTure(cfg)

    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        print('trainer.paint!');
        trainer.paint()


if __name__ == '__main__':
    main()
