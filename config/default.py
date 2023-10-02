import os, sys
import random
import string
from yacs.config import CfgNode as CN

cfg = CN()
cfg.SEED = 42
cfg.run_key = ''.join(random.sample(string.ascii_lowercase, 8))

cfg.base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
cfg.data_path = '/home/guests/alexander_baumann/data'
cfg.output_path = '/home/guests/alexander_baumann/out'

cfg.data = CN()
cfg.data.data_folds = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
]

# which fold to use for val, use rest for train
cfg.data.val_fold = 3
# even indices are right thyroid lobes
# i.e. {0, 2, ... 30}
cfg.data.side = "right"

# all the SSM configuration values
cfg.ssm = CN()
cfg.ssm.path_prefix = 'surfmnet'
cfg.ssm.weights = None  # '../data/save/2022_11_05_09_29_43/model_last.pth'T
cfg.ssm.dataroot = './data/'
cfg.ssm.info = 'training surfmnet gnn femur noisy det 2 '
cfg.ssm.dim_basis_model = 20
cfg.ssm.feat_size = 352
cfg.ssm.output_dir = "./out"
cfg.ssm.num_eigen = 20
cfg.ssm.njobs = 4
cfg.ssm.test = False
cfg.ssm.sinkhorn = False
cfg.ssm.arch = 'resnet'
cfg.ssm.TrainingDetails = CN()
cfg.ssm.TrainingDetails.n_epochs = 30
cfg.ssm.TrainingDetails.save = True
cfg.ssm.TrainingDetails.n_sample_points = 2048
cfg.ssm.TrainingDetails.lr = 0.0001
cfg.ssm.TrainingDetails.batch_size = 64
cfg.ssm.shot = True
cfg.ssm.n_points = 4000
cfg.ssm.TestingDetails = CN()
cfg.ssm.TestingDetails.n_sample_points = None
cfg.ssm.TestingDetails.ref_thyroid = 0
cfg.ssm.TestingDetails.landmarks = False
cfg.ssm.TestingDetails.batch_size = 1



def get_cfg_defaults():
    return cfg.clone()
