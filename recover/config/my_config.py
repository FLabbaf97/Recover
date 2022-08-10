from recover.my_codes.my_model import MyModel, MyTrainer, MyBilinearMLPPredictor, NormalMergeModel
from recover.datasets.drugcomb_matrix_data import DrugCombMatrix
from recover.models.models import Baseline
from recover.models.predictors import BilinearFilmMLPPredictor, BilinearMLPPredictor
from recover.utils.utils import get_project_root
from recover.train import train_epoch, eval_epoch, BasicTrainer
import os
from ray import tune
from importlib import import_module

########################################################################################################################
# Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": False,  # Unchanged
    "num_epoch_without_tune": 1000,  # Used only if "use_tune" == False
    # "seed": tune.grid_search([2, 3, 4]), # Unchanged
    "seed": 0,
    # Optimizer config
    "lr": 1e-4,  # Unchanged
    "weight_decay": 1e-2,  # Unchanged
    "batch_size": 128,  # Unchanged
    # Train epoch and eval_epoch to use
    "train_epoch": train_epoch,  # Unchanged
    "eval_epoch": eval_epoch,  # Unchanged
}

predictor_config = {
    "predictor": NormalMergeModel,
    "predictor_layers":
    [
        1024,
        256,
        128,
        64,
        1,
    ],
    # Computation on the sum of the two drug embeddings for the last n layers
    "merge_n_layers_before_the_end": 5,
    "allow_neg_eigval": True,  # Unchanged
}

model_config = {
    "model": MyModel,  # Unchanged yet, maybe I should change it.
    "load_model_weights": False,  # Unchanged yet. where can I load model weights??
}


# list of cell lines, but I dont use it right now
"""
List of cell line names: 

['786-0', 'A498', 'A549', 'ACHN', 'BT-549', 'CAKI-1', 'EKVX', 'HCT-15', 'HCT116', 'HOP-62', 'HOP-92', 'HS 578T', 'HT29',
 'IGROV1', 'K-562', 'KM12', 'LOX IMVI', 'MALME-3M', 'MCF7', 'MDA-MB-231', 'MDA-MB-468', 'NCI-H226', 'NCI-H460', 
 'NCI-H522', 'NCIH23', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'OVCAR3', 'PC-3', 'RPMI-8226', 'SF-268', 'SF-295', 'SF-539', 
 'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'SK-OV-3', 'SNB-75', 'SR', 'SW-620', 'T-47D', 'U251', 'UACC-257', 'UACC62', 
 'UO-31']
"""

dataset_config = {  # Unchanged
    "dataset": DrugCombMatrix,
    "study_name": 'ALMANAC',
    "in_house_data": 'without',
    "rounds_to_include": [],
    "val_set_prop": 0.2,
    "test_set_prop": 0.1,
    "test_on_unseen_cell_line": False,
    "split_valid_train": "pair_level",
    "cell_line": 'MCF7',  # 'PC-3',
    # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
    "target": "bliss_max",
    "fp_bits": 1024,
    "fp_radius": 2
}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "trainer": MyTrainer,  # PUT NUM GPU BACK TO 1
    "trainer_config": {
        **pipeline_config,
        **predictor_config,
        **model_config,
        **dataset_config,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "stop": {"training_iteration": 1000, 'patience': 10},
    "checkpoint_score_attr": 'eval/comb_r_squared',
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 1,
    "resources_per_trial": {"cpu": 8, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}
