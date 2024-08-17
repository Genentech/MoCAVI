"""Utilities for contrastiveVI modules."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from scvi.nn import FCLayers
from torch.distributions import Normal


def product_of_experts(mus, vars):
    mus_joint = torch.sum(mus / vars, dim=-1)  # batch size
    vars_joint = torch.ones_like(mus_joint) + torch.sum(1.0 / vars, dim=-1)
    vars_joint = 1.0 / vars_joint  # inverse
    mus_joint *= vars_joint
    return mus_joint, vars_joint


class DecoderADT(torch.nn.Module):
    """Decoder for just surface proteins (ADT)."""

    def __init__(
        self,
        n_input: int,
        n_output_proteins: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()
        self.n_output_proteins = n_output_proteins

        linear_args = {
            "n_layers": 1,
            "use_activation": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0,
        }

        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=nn.ReLU,
        )

        self.py_background_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # dropout (mixture component for proteins, ZI probability for genes)
        self.sigmoid_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # background mean parameters second decoder
        self.py_back_mean_log_alpha = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        self.py_back_mean_log_beta = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        """Forward pass."""
        # z is the latent repr
        py_ = {}

        py_back = self.py_back_decoder(z, *cat_list)
        py_back_cat_z = torch.cat([py_back, z], dim=-1)

        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z, *cat_list)
        py_["back_beta"] = torch.exp(self.py_back_mean_log_beta(py_back_cat_z, *cat_list))
        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        py_fore = self.py_fore_decoder(z, *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = self.py_fore_scale_decoder(py_fore_cat_z, *cat_list) + 1 + 1e-8
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        p_mixing = self.sigmoid_decoder(z, *cat_list)
        p_mixing_cat_z = torch.cat([p_mixing, z], dim=-1)
        py_["mixing"] = self.py_background_decoder(p_mixing_cat_z, *cat_list)

        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        py_["scale"] = torch.nn.functional.normalize((1 - protein_mixing) * py_["rate_fore"], p=1, dim=-1)

        return py_, log_pro_back_mean
