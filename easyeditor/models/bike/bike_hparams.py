from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class BIKEHyperParams(HyperParams):
    # Method
    model_name: str
    results_dir: str
    device: int
    alg_name: str
    max_length: int
    weight_f: float
    weight_g: float 
    weight_patch: float
    model_parallel: bool = False
    fp16: bool = False

    
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'BIKE') or print(f'BIKEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
