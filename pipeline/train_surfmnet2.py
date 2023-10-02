import sys
import string
import random
import yaml
from pathlib import Path
import time
import torch
sys.path.append('..')
# import convert_CN_to_dict
from config import convert_CN_to_dict
from pipeline.p_utils.pipeline_utils import parse_config, get_train_val_sets, seed_everything, get_train_val_sets_normal

# load surfmnet paths
sys.path.append('../Statistical Thyroid Model/Functional_maps_approach/surfmnet')
sys.path.append('../Statistical Thyroid Model/Functional_maps_approach/surfmnet/utils')

from ssm_pipeline import ssm_pipeline


def train_ssm(config):
    config.freeze()
    print("--- Using configs ---")
    print(config)
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_key = config.run_key
    seed_everything(config.SEED)

    val_fold = config.data.val_fold
    train_set, val_set = get_train_val_sets_normal(config, val_fold)
    naming = str(config.data_path).split('/')[-1]
    output_path = Path(config.output_path) / config.ssm.path_prefix / f'experiment_fold_{val_fold}_{run_key}_{naming}'
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        yaml_file = output_path / 'pipeline_config.yml'
        with yaml_file.open('w') as f:
            yaml.dump(convert_CN_to_dict(config, []), f)
    train_paths = ['{:s}/{:03d}.ply'.format(config.data_path,i) for i in train_set]
    val_paths = ['{:s}/{:03d}.ply'.format(config.data_path,i) for i in val_set]
    # test generates the outputs.. use both train and val
    test_paths = train_paths + val_paths
    args = convert_CN_to_dict(config.ssm, [])
    print(args)
    # first train surfmnet correspondence pipeline
    mat_savedir, ref_thyroid = ssm_pipeline(args, train_paths, val_paths, test_paths,
                                            save_path=output_path)
    print(f"Saved correspondence files to {mat_savedir} for reference thyroid {ref_thyroid}")


if __name__ == "__main__":
    cfg = parse_config()
    for i in range(4):
        t = time.time()
        cfg.defrost()
        cfg.data.val_fold = i
        train_ssm(cfg)
        print((time.time()-t)/3600)
