import sys
from pathlib import Path
from pipeline.p_utils.pipeline_utils import get_train_val_sets
from pipeline.p_utils.pipeline_utils import parse_config, seed_everything
from pipeline.p_utils.pipeline_utils import get_correspondence_files
from pipeline.p_utils.pipeline_utils import get_ssm_cache_dir

sys.path.append("./Partial Registration/")
sys.path.append('./Statistical Thyroid Model/Functional_maps_approach/')
sys.path.append("./Partial Registration/SdfVoxelMatching/")

from generate_dataset_crossval import SSMsample2sdf


def generate_ssm_cache(output_path, config, train_files, val_fold):
    """
    Generate a cache for the SSM samples for a specific fold and
    level of variance.
    """
    std = config.ssm_cache.std
    cache_dir = get_ssm_cache_dir(config, output_path, val_fold, std)
    cache_dir.mkdir(parents=True, exist_ok=True)
    N_samples = config.ssm_cache.max_samples
    SSMsample2sdf(N_samples, train_files, cache_dir, std=std)


if __name__ == "__main__":
    config = parse_config()
    run_key = config.run_key
    seed_everything(config.SEED)
    val_fold = config.data.val_fold
    train_indices, val_indices = get_train_val_sets(config, val_fold)

    corr_files = config.reg.corr_files
    base_path = Path(config.output_path) / config.ssm.path_prefix
    train_files = get_correspondence_files(base_path, corr_files, val_fold, train_indices)

    # create directories where to sample SSM files
    # pass directly to train after
    generate_ssm_cache(Path(config.output_path), config.reg, train_files, val_fold)
