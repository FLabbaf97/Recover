import torch
import torch.nn as nn
from torch.nn import Parameter

########################################################################################################################
# Modules
########################################################################################################################


class LinearModule(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearModule, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ReLUModule(nn.ReLU):
    def __init__(self):
        super(ReLUModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class FilmModule(torch.nn.Module):
    def __init__(self, num_cell_lines, out_dim):
        super(FilmModule, self).__init__()
        film_init = 1 / 100 * torch.randn(num_cell_lines, 2 * out_dim)
        film_init = film_init + torch.Tensor([([1] * out_dim) + ([0] * out_dim)])

        self.film = Parameter(film_init)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [
            self.film[cell_line][:, : x.shape[1]] * x
            + self.film[cell_line][:, x.shape[1]:],
            cell_line]


class FilmWithFeatureModule(torch.nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(FilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, num_cell_line_features)
        self.condit_relu = nn.ReLU()
        self.condit_lin_2 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_2.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_2(self.condit_relu(self.condit_lin_1(cell_line_features)))

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]


class LinearFilmWithFeatureModule(torch.nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(LinearFilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_1.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_1(cell_line_features)

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]


########################################################################################################################
# Bilinear MLP
########################################################################################################################


class BilinearMLPPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(BilinearMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # Build last layers (after addition of the two embeddings)
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

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

        self.allow_neg_eigval = config["allow_neg_eigval"]
        if self.allow_neg_eigval:
            self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

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

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


########################################################################################################################
# Bilinear MLP with Film conditioning
########################################################################################################################


class BilinearFilmMLPPredictor(BilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearFilmMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]


class BilinearFilmWithFeatMLPPredictor(BilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self. cl_features_dim = data.cell_line_features.shape[1]
        super(BilinearFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features


class BilinearLinFilmWithFeatMLPPredictor(BilinearFilmWithFeatMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearLinFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), LinearFilmWithFeatureModule(self. cl_features_dim,
                                                                               self.layer_dims[i + 1])]


########################################################################################################################
# Bilinear MLP with Cell line features as input
########################################################################################################################


class BilinearCellLineInputMLPPredictor(BilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self.cl_features_dim = data.cell_line_features.shape[1]
        predictor_layers[0] += self.cl_features_dim
        super(BilinearCellLineInputMLPPredictor, self).__init__(data, config, predictor_layers)

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        # Add cell line features to drug representations
        h_drug_1 = torch.cat((h_drug_1, batch_cl_features), dim=1)
        h_drug_2 = torch.cat((h_drug_2, batch_cl_features), dim=1)

        return h_drug_1, h_drug_2, cell_lines