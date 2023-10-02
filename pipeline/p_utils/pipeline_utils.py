import sys
import random
from pathlib import Path
from copy import deepcopy
from natsort import natsorted

import torch
import numpy as np

from config import get_cfg_defaults


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_config():
    """
    parse the config file. Merge from an optional
    file if provided (see yacs lib).
    override with additional pairs of args from command
    line.
    Sample usage:
    `PYTHONPATH=./ python pipeline/train_2d3d.py config/debug.yml data.val_fold 3`
    """
    cfg = get_cfg_defaults()
    if len(sys.argv) > 1:
        cfg.merge_from_file(str(sys.argv[1]))

    if len(sys.argv) > 2:
        params = sys.argv[2:]
        out = []
        for param in params:
            # if '=' in param:
            #     # other param format with arg=val
            #     out.extend(param.split("="))
            # else:
            out.append(param)
        cfg.merge_from_list(out)
    return cfg


def get_train_val_sets(config, val_fold):
    """
    get the traina and validation splits based
    on the fold. Make sure they don't overlap
    Returns two arrays.
    """
    data = deepcopy(config.data.data_folds)
    val_set = np.array(data.pop(val_fold))
    train_set = np.concatenate(data)
    assert len(np.intersect1d(val_set, train_set)) == 0

    train_set *= 2
    val_set *= 2
    if config.data.side == "left":
        train_set += 1
        val_set += 1
        assert np.all(train_set % 2 == 1)
    elif config.data.side == "right":
        assert np.all(train_set % 2 == 0)

    return train_set, val_set

def get_train_val_sets_normal(config, val_fold):
    """
    get the traina and validation splits based
    on the fold. Make sure they don't overlap
    Returns two arrays.
    """
    data = deepcopy(config.data.data_folds)
    val_set = np.array(data.pop(val_fold))
    train_set = np.concatenate(data)
    # assert len(np.intersect1d(val_set, train_set)) == 0

    return train_set, val_set


def get_correspondence_files(base_path, corr_files, val_fold, indices):
    """
    Get the .mat correspondence files.
    we don't double check that the reference is correct..
    We do check that it is from the correct validation fold.

    files should have the structure:
    `out_x_n.mat`
    where x is the index of the reference thyroid
    and will be filtered based on n present in the `indices` list
    """
    # make sure params are set correctly
    assert corr_files is not None
    assert f"fold_{val_fold}" in corr_files, f"fold_{val_fold} not in {corr_files}"
    # specific structure for the output files from surfmnet
    corr_dir = base_path / corr_files / "mat"
    train_files = [
        x for x in
        # we don't double check that these are correct..
        corr_dir.glob('out_*.mat')
        if int(x.stem.split('_')[2].split('.')[0]) in indices
    ]
    train_files = natsorted(train_files, key=lambda x: x.name)
    assert len(train_files) == len(indices), f"train files {train_files} do not match indicies {indices}"
    for p in train_files:
        assert p.exists(), f"File {p} does not exist"
    return train_files


def get_ssm_cache_dir(config, output_path, val_fold, std):
    """pass in config.reg"""
    return Path(output_path) / config.ssm_cache.dir / f"val_fold_{val_fold}_std_{std}"

