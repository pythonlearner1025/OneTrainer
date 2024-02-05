import json
from abc import ABCMeta

import torch
from mgds.MGDS import MGDS

from modules.util.TrainProgress import TrainProgress
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
    ):
        if config.concepts is not None:
            concepts = config.concepts
        else:
            with open(config.concept_file_name, 'r') as f:
                concepts = json.load(f)
                for i in range(len(concepts)):
                    concepts[i] = ConceptConfig.default_values().from_dict(concepts[i]).to_dict()

        settings = {
            "enable_random_circular_mask_shrink": config.circular_mask_generation,
            "enable_random_mask_rotate_crop": config.random_rotate_and_crop,
            "target_resolution": config.resolution,
        }

        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size,
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
