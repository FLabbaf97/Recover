import torch
from torch.nn import Parameter
import numpy as np
import os
from torch import device
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ray import tune
# import ray
# import time
# import argparse
# import importlib
from scipy import stats
from scipy.stats import spearmanr

from recover.utils.utils import get_tensor_dataset


class MyTrainer(tune.Trainable):
    def setup(self, config):
        print("Initializing regular training pipeline with MyTrainer")

        print("Initializing regular training pipeline with MyTrainer")

        self.batch_size = config["batch_size"]
        device_type = "cpu"
        print("device_type: ", device_type)
        self.device = torch.device(device_type)
        self.training_it = 0

        # Initialize dataset
        dataset = config["dataset"](
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            cell_line=config["cell_line"],
            study_name=config["study_name"],
            in_house_data=config["in_house_data"],
            rounds_to_include=config.get("rounds_to_include", []),
        )

        self.data = dataset.data.to(self.device)

        # If a score is the target, we store it in the ddi_edge_response attribute of the data object
        if "target" in config.keys():
            possible_target_dicts = {
                "bliss_max": self.data.ddi_edge_bliss_max,
                "bliss_av": self.data.ddi_edge_bliss_av,
                "css_av": self.data.ddi_edge_css_av,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        # Perform train/valid/test split. Test split is fixed regardless of the user defined seed
        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(
            config)

        # Train loader
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Valid loader
        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Initialize model
        self.model = config["model"](self.data, config)

	#TODO: see how to save weights and load them again
        # # Initialize model with weights from file
        # load_model_weights = config.get("load_model_weights", False)
        # if load_model_weights:
        #     model_weights_file = config.get("model_weights_file")
        #     model_weights = torch.load(model_weights_file, map_location="cpu")
        #     self.model.load_state_dict(model_weights)
        #     print("pretrained weights loaded")
        # else:
        #     print("model initialized randomly")

        self.model = self.model.to(self.device)
        print(self.model)

        # Initialize optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        self.patience = 0
        self.max_eval_r_squared = -1

    def step(self):
        print("Initializing regular training pipeline with MyTrainer")
        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
        )

        eval_metrics, _ = self.eval_epoch(
            self.data, self.valid_loader, self.model)

        # print("eval metrics: ", eval_metrics)
        # print("train metrics: ", train_metrics)

        train_metrics = [("train/" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)
        # print("metrics: ", metrics)

        metrics["training_iteration"] = self.training_it
        self.training_it += 1

        # Compute patience
        if metrics['eval/comb_r_squared'] > self.max_eval_r_squared:
            self.patience = 0
            self.max_eval_r_squared = metrics['eval/comb_r_squared']
        else:
            self.patience += 1

        metrics['patience'] = self.patience
        metrics['all_space_explored'] = 0

        return metrics

    # def save_checkpoint(self, checkpoint_dir):
    #     checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    #     torch.save(self.model.state_dict(), checkpoint_path)
    #     return checkpoint_path

    # def load_checkpoint(self, checkpoint_path):
    #     self.model.load_state_dict(torch.load(checkpoint_path))


class MyModel(torch.nn.Module):
    def __init__(self, data, config):

        super(MyModel, self).__init__()

        device_type = 'cpu'  # TODO: change this to mps later
        # device_type = "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.criterion = torch.nn.MSELoss()

        # Compute dimension of input for predictor
        predictor_layers = [2*data.x_drugs.shape[1]] + \
            config["predictor_layers"]

        assert predictor_layers[-1] == 1

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def loss(self, output, drug_drug_batch):
        """
        Loss function for the synergy prediction pipeline
        :param output: output of the predictor
        :param drug_drug_batch: batch of drug-drug combination examples
        :return:
        """
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)

        return loss


class MyBilinearMLPPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(MyBilinearMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        print("number of cell_lines: ", self.num_cell_lines)
        # device_type = "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
        device_type = "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        # layers_before_merge = []
        # layers_after_merge = []

        # # Build early layers (before addition of the two embeddings)
        # for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
        #     layers_before_merge = self.add_layer(
        #         layers_before_merge,
        #         i,
        #         self.layer_dims[i],
        #         self.layer_dims[i + 1]
        #     )

        # # Build last layers (after addition of the two embeddings)
        # for i in range(
        #     len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
        #     len(self.layer_dims) - 1,
        # ):

        #     layers_after_merge = self.add_layer(
        #         layers_after_merge,
        #         i,
        #         self.layer_dims[i],
        #         self.layer_dims[i + 1]
        #     )

        # self.before_merge_mlp = torch.nn.Sequential(*layers_before_merge)
        # self.after_merge_mlp = torch.nn.Sequential(*layers_after_merge)

        self.before_merge_mlp = self.build_before_merge_layers()
        self.after_merge_mlp = self.build_after_merge_layers()

        self.allow_neg_eigval = config["allow_neg_eigval"]

        # self.bilinear_weights, self.bilinear_offsets, self.bilinear_diag = self.build_bilinear_tensor()

    def build_bilinear_tensor(self):
        print("biulding bilinear tensor")

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        ## how: torch.eye Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
        ## produce 1024 (#self.merge_dim) of the above 2-D tensor and cat them together to crease a 1024*1024*1024 tensor wich each 2-D is identical
        ## sum the above tensor with 1/100 of a random from normal distribution

        bilinear_weights = Parameter(
            1 / 100 *
            torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]]
                        * self.merge_dim, dim=0)
        )
        print("1111111111111111")
        bilinear_offsets = Parameter(
            1 / 100 * torch.randn((self.merge_dim)))
        print("22222222222222222222")
        if self.allow_neg_eigval:
            bilinear_diag = Parameter(
                1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)
        print("333333333333333333")
        return bilinear_weights, bilinear_offsets, bilinear_diag

    def build_before_merge_layers(self):
        print("biulding before merge layers")

        layers_before_merge = []
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )
        return torch.nn.Sequential(*layers_before_merge)

    def build_after_merge_layers(self):
        print("biulding after merge layers")

        layers_after_merge = []
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):

            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )
        return torch.nn.Sequential(*layers_after_merge)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        print("before merge completed successfully")

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        if self.allow_neg_eigval:
            # Multiply by diagonal matrix to allow for negative eigenvalues
            h_2 *= self.bilinear_diag

        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)

        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)

        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets
        print("bilinear merge completed successfully")

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        # Cell line of all examples in the batch
        cell_lines = drug_drug_batch[1]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, activation=None):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if(activation):
            layers.append(activation())

        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


class LinearModule(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearModule, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ReLUModule(torch.nn.ReLU):
    def __init__(self):
        super(ReLUModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class NormalMergeModel(MyBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(NormalMergeModel, self).__init__(
            data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = h_drug_1
        h_2 = h_drug_2

        h_1_scal_h_2 = torch.cat((h_1, h_2), 1)

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def build_after_merge_layers(self):
        layers_after_merge = []
        # first after merge layer dim is twice bigger than previous layer dimention
        current_layer = 0
        for i in range(
            len(self.layer_dims) - 1,
        ):

            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )
        return torch.nn.Sequential(*layers_after_merge)

    # def linear_layer(self, i, dim_i, dim_i_plus_1):
    #     print("Add layer: %d and %d dimentions", dim_i, dim_i_plus_1)
    #     return [LinearModule(dim_i, dim_i_plus_1)]

    # def add_layer(self, layers, i, dim_i, dim_i_plus_1, activation=None):
    #     layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
    #     if(activation):
    #         layers.append(activation())

    #     if i != len(self.layer_dims) - 2:
    #         layers.append(ReLUModule())

    #     return layers
