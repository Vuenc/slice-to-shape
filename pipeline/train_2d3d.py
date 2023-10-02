import yaml
import sys
from natsort import natsorted
from pathlib import Path

import torch

from config import convert_CN_to_dict
from pipeline.p_utils.pipeline_utils import parse_config, seed_everything
from pipeline.p_utils.pipeline_utils import get_train_val_sets, get_correspondence_files
from pipeline.p_utils.pipeline_utils import get_ssm_cache_dir

sys.path.append('./Statistical Thyroid Model/Functional_maps_approach/')
sys.path.append("./Partial Registration/SdfVoxelMatching/")
sys.path.append("./Partial Registration/SdfVoxelMatching/pytorch-3dunet/")
sys.path.append("./Partial Registration/")

from generate_dataset_crossval import SSMsample2sdf, SSMmean2sdf
from ssm_thyroid_dataset import SSMvoxelDataset
from train_ssmvoxel import train


def train_registration(config, data_dir, sdf_dir_train, sdf_dir_val, train_idx, val_idx, output_path):
    us_paths_train = natsorted([Path(data_dir) / f"{us_index}.nii"
                                for us_index in train_idx],
                               key=lambda x: x.name)
    us_paths_val = natsorted([Path(data_dir) / f"{us_index}.nii"
                              for us_index in val_idx],
                             key=lambda x: x.name)

    sdf_paths_train = natsorted([x for x in sdf_dir_train.glob("SSMsample_*_sdf")],
                                key=lambda x: x.name)
    print(f"Found {len(sdf_paths_train)} SSM samples in {sdf_dir_train}")
    sdf_paths_val = natsorted([x for x in sdf_dir_val.glob("SSMsample_*_sdf")],
                              key=lambda x: x.name)
    # only pass single mean shape. should always be 1
    assert len(sdf_paths_val) == 1

    train_dataset = SSMvoxelDataset(us_paths_train, sdf_paths_train,
                                    config.N_patches_per_thyroid,
                                    sdf_dir=sdf_dir_train,
                                    ultrasound_patches_shape=config.US_patch_shape,
                                    sdf_patches_shape=config.sdf_patch_shape,
                                    negative_sample_from_p_furthest=config.negative_sample_dist)
    val_dataset = SSMvoxelDataset(us_paths_val, sdf_paths_val,
                                  config.N_patches_per_thyroid,
                                  sdf_dir=sdf_dir_val,
                                  ultrasound_patches_shape=config.US_patch_shape,
                                  sdf_patches_shape=config.sdf_patch_shape,
                                  negative_sample_from_p_furthest=config.negative_sample_dist)
    torch.cuda.empty_cache()
    train(train_dataset=train_dataset, val_dataset=val_dataset,
          epochs=config.train.epochs, device="cuda",
          reshuffle_samples=True,
          batch_size=config.train.batch_size, tensorboard=True,
          loss_type='weightedSoftMarginTriplet',
          data_parallel_gpu_ids=config.train.gpu_ids,
          stopping_patience=0,
          save_dir=output_path / "saves",
          log_dir=output_path / "runs",
          loss_params={"alpha": config.train.alpha},
          optimizer_params={"lr": config.train.lr},
          num_workers=config.njobs, pin_memory=True)


def sample_ssm(config, train_files, val_files, experiment_path):
    train_dir = None
    val_dir = experiment_path / "meansdf_val"
    if config.reg.mode == "ssm_samples":
        std = config.reg.ssm_sample_std
        val_fold = config.data.val_fold
        cache_dir = get_ssm_cache_dir(config.reg, config.output_path, val_fold, std)
        print(f"Sampling SSM and writing files to {train_dir}")
        # SSMsample2sdf(N_samples, train_files, train_dir, std=std)
        assert cache_dir.exists(), f"SSM cache dir {cache_dir} does not exist!"
        train_dir = cache_dir
    else:
        # sdf dir is set to mean dir for this training mode
        # this means we train and validation on mean shape only
        train_dir = experiment_path / "meansdf_train"
        train_dir.mkdir(parents=True, exist_ok=True)
        SSMmean2sdf(train_files, train_files, train_dir)

    # validation is always the same, with mean shape
    val_dir.mkdir(parents=True, exist_ok=True)
    SSMmean2sdf(train_files, val_files, val_dir)
    return train_dir, val_dir


if __name__ == "__main__":
    config = parse_config()
    run_key = config.run_key
    seed_everything(config.SEED)
    val_fold = config.data.val_fold
    train_indices, val_indices = get_train_val_sets(config, val_fold)

    corr_files = config.reg.corr_files
    base_path = Path(config.output_path) / config.ssm.path_prefix
    train_files = get_correspondence_files(base_path, corr_files,
                                           val_fold, train_indices)
    val_files = get_correspondence_files(base_path, corr_files,
                                         val_fold, val_indices)

    # create directories where to sample SSM files
    # pass directly to train after
    experiment_path = Path(config.output_path) / config.reg.path_prefix / f"sdf_train_fold_{val_fold}_{run_key}"
    if not experiment_path.exists():
        experiment_path.mkdir(parents=True, exist_ok=True)
        yaml_file = experiment_path / 'pipeline_config.yml'
        with yaml_file.open('w') as f:
            yaml.dump(convert_CN_to_dict(config, []), f)

    # get the train_sdf_paths, and val_sdf_paths
    # depending on training mode
    sdf_dir_train, sdf_dir_val = sample_ssm(config, train_files, val_files, experiment_path)

    train_registration(config.reg, config.data_path, sdf_dir_train,
                       sdf_dir_val, train_indices, val_indices, experiment_path)
