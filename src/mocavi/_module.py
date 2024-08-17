"""PyTorch module for MoCAVI for single cell multi-omics data."""

from collections.abc import Iterable

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture, Poisson
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ._utils import DecoderADT, product_of_experts

torch.backends.cudnn.benchmark = True


class MoCAVIModule(BaseModuleClass):
    """
    PyTorch module for MoCAVI.

    Args:
    ----
        n_input: Number of input genes.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        disentanglement_penalty: Weight of the disentanglement loss that further
            discourages shared variations from leaking into the salient latent space.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_regions: int,
        n_input_proteins: int,
        n_labels: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        protein_background_prior_mean: np.ndarray | None = None,
        protein_background_prior_scale: np.ndarray | None = None,
        wasserstein_reg: float = 0,
    ) -> None:
        super().__init__()

        self.n_input_regions = n_input_regions
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.n_hidden = n_hidden

        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.dropout_rate = dropout_rate

        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        n_total_latent = n_background_latent + n_salient_latent
        self.wasserstein_reg = wasserstein_reg

        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else [n_batch]

        ## Initialize tensors and neural nets

        # deal with the case where there is no input for a given modality
        input_exp = 1 if self.n_input_genes == 0 else self.n_input_genes
        input_acc = 1 if self.n_input_regions == 0 else self.n_input_regions
        input_pro = 1 if self.n_input_proteins == 0 else self.n_input_proteins

        # expression encoder z
        self.z_encoder_expression = Encoder(
            n_input=input_exp,
            n_output=self.n_background_latent,
            n_cat_list=None,
            n_layers=self.n_layers_encoder,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            distribution="normal",
            inject_covariates=False,
            use_batch_norm=False,
            use_layer_norm=True,
            activation_fn=torch.nn.LeakyReLU,
            var_eps=0,
            return_dist=False,
        )
        # expression encoder s
        self.s_encoder_expression = Encoder(
            n_input=input_exp,
            n_output=self.n_salient_latent,
            n_cat_list=None,
            n_layers=self.n_layers_encoder,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            distribution="normal",
            inject_covariates=False,
            use_batch_norm=False,
            use_layer_norm=True,
            activation_fn=torch.nn.LeakyReLU,
            var_eps=0,
            return_dist=False,
        )
        # expression decoder
        self.decoder_expression = DecoderSCVI(
            n_total_latent,
            n_input_genes,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=self.n_hidden,
            inject_covariates=True,
            use_batch_norm=False,
            use_layer_norm=True,
            scale_activation="softmax",
        )
        # expression dispersion
        self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))

        # accessibility
        # accessibility encoder z
        self.z_encoder_accessibility = Encoder(
            n_input=input_acc,
            n_layers=self.n_layers_encoder,
            n_output=self.n_background_latent,
            n_hidden=self.n_hidden,
            n_cat_list=None,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution="normal",
            var_eps=0,
            use_batch_norm=False,
            use_layer_norm=True,
            return_dist=False,
        )
        # accessibility encoder s
        self.s_encoder_accessibility = Encoder(
            n_input=input_acc,
            n_layers=self.n_layers_encoder,
            n_output=self.n_salient_latent,
            n_hidden=self.n_hidden,
            n_cat_list=None,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution="normal",
            var_eps=0,
            use_batch_norm=False,
            use_layer_norm=True,
            return_dist=False,
        )
        # accessibility decoder
        self.decoder_accessibility = DecoderSCVI(
            n_total_latent,
            input_acc,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=self.n_hidden,
            inject_covariates=True,
            use_batch_norm=False,
            use_layer_norm=True,
            scale_activation="softmax",
        )
        # protein encoder z
        self.z_encoder_protein = Encoder(
            n_input=input_pro,
            n_layers=self.n_layers_encoder,
            n_output=self.n_background_latent,
            n_hidden=self.n_hidden,
            n_cat_list=None,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution="normal",
            var_eps=0,
            use_batch_norm=False,
            use_layer_norm=True,
            return_dist=False,
        )
        # protein encoder s
        self.s_encoder_protein = Encoder(
            n_input=input_pro,
            n_layers=self.n_layers_encoder,
            n_output=self.n_salient_latent,
            n_hidden=self.n_hidden,
            n_cat_list=None,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution="normal",
            var_eps=0,
            use_batch_norm=False,
            use_layer_norm=True,
            return_dist=False,
        )

        # protein decoder
        self.decoder_pro = DecoderADT(
            n_input=n_total_latent,
            n_output_proteins=n_input_proteins,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=False,
            use_layer_norm=True,
            deep_inject_covariates=True,
        )

        # protein dispersion parameters
        self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))

        ## extra parameters or tensors
        # weights for reweighting strategies
        self.register_buffer("background_weight", torch.tensor(0.5, dtype=torch.float32))
        # chem specific priors
        self.label_specific_prior_mean = torch.nn.parameter.Parameter(
            0.01 * torch.randn((self.n_labels, self.n_salient_latent))
        )

    @staticmethod
    def _get_min_batch_size(concat_tensors: dict[str, dict[str, torch.Tensor]]) -> int:
        return min(
            concat_tensors["background"][REGISTRY_KEYS.X_KEY].shape[0],
            concat_tensors["target"][REGISTRY_KEYS.X_KEY].shape[0],
        )

    @staticmethod
    def _reduce_tensors_to_min_batch_size(tensors: dict[str, torch.Tensor], min_batch_size: int) -> None:
        for name, tensor in tensors.items():
            tensors[name] = tensor[:min_batch_size, :]

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        concat_tensors: dict[str, dict[str, torch.Tensor]], index: str
    ) -> dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = {"x": x, "y": y, "batch_index": batch_index}
        return input_dict

    def _get_inference_input(
        self, concat_tensors: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(concat_tensors, "background")
        target = self._get_inference_input_from_concat_tensors(concat_tensors, "target")
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target, min_batch_size)
        return {"background": background, "target": target}

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        # Get Data and Additional Covs
        dummy_data = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        if self.n_input_genes == 0:
            x_rna = dummy_data
        else:
            x_rna = x[:, : self.n_input_genes]

        if self.n_input_regions == 0:
            x_chr = dummy_data
        else:
            x_chr = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]

        if self.n_input_proteins == 0:
            y = dummy_data

        # encode expression
        library_rna = torch.log(x_rna.sum(1)).unsqueeze(1)
        x_rna_log = torch.log(1 + x_rna)
        qz_exp_m, qz_exp_v, z_exp = self.z_encoder_expression(x_rna_log, batch_index)
        qs_exp_m, qs_exp_v, s_exp = self.s_encoder_expression(x_rna_log, batch_index)

        # encode chromatin
        library_acc = torch.log(x_chr.sum(1)).unsqueeze(1)
        x_chr_log = torch.log(1 + x_chr)
        qz_chr_m, qz_chr_v, z_chr = self.z_encoder_accessibility(x_chr_log, batch_index)
        qs_chr_m, qs_chr_v, s_chr = self.s_encoder_accessibility(x_chr_log, batch_index)

        # encode proteins
        y_log = torch.log(1 + y)
        qz_pro_m, qz_pro_v, z_pro = self.z_encoder_protein(y_log, batch_index)
        qs_pro_m, qs_pro_v, s_pro = self.s_encoder_protein(y_log, batch_index)

        # combine latent variables from multiple modalities in order to create the meta-variable.
        z_mus = torch.stack([qz_exp_m, qz_chr_m, qz_pro_m], dim=-1)
        z_vars = torch.stack([qz_exp_v, qz_chr_v, qz_pro_v], dim=-1)
        s_mus = torch.stack([qs_exp_m, qs_chr_m, qs_pro_m], dim=-1)
        s_vars = torch.stack([qs_exp_v, qs_chr_v, qs_pro_v], dim=-1)
        qz_m, qz_v = product_of_experts(z_mus, z_vars)
        qs_m, qs_v = product_of_experts(s_mus, s_vars)
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()
        s = Normal(qs_m, torch.sqrt(qs_v)).rsample()

        trace_tensors = [
            qz_m,
            qz_v,
            z,
            qs_m,
            qs_v,
            s,
            qz_exp_m,
            qz_exp_v,
            z_exp,
            qs_exp_m,
            qs_exp_v,
            s_exp,
            qz_chr_m,
            qz_chr_v,
            z_chr,
            qs_chr_m,
            qs_chr_v,
            s_chr,
            qz_pro_m,
            qz_pro_v,
            z_pro,
            qs_pro_m,
            qs_pro_v,
            s_pro,
            library_rna,
            library_acc,
        ]
        trace_names = [
            "qz_m",
            "qz_v",
            "z",
            "qs_m",
            "qs_v",
            "s",
            "qz_exp_m",
            "qz_exp_v",
            "z_exp",
            "qs_exp_m",
            "qs_exp_v",
            "s_exp",
            "qz_chr_m",
            "qz_chr_v",
            "z_chr",
            "qs_chr_m",
            "qs_chr_v",
            "s_chr",
            "qz_pro_m",
            "qz_pro_v",
            "z_pro",
            "qs_pro_m",
            "qs_pro_v",
            "s_pro",
            "library_rna",
            "library_acc",
        ]

        outputs = {}
        if n_samples == 1:
            for i, tensor in enumerate(trace_tensors):
                outputs[trace_names[i]] = tensor
        else:
            for i, tensor in enumerate(trace_tensors):
                outputs[trace_names[i]] = self._reshape_tensor_for_samples(tensor, n_samples)
            # resample with the right number of samples
            outputs["z"] = Normal(outputs["qz_m"], torch.sqrt(outputs["qz_v"])).rsample()
            outputs["s"] = Normal(outputs["qs_m"], torch.sqrt(outputs["qs_v"])).rsample()

        return outputs

    @auto_move_data
    def inference(
        self,
        background: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # collate all tensors together and pass through inference networks
        background_batch_size = background["x"].shape[0]
        target_batch_size = target["x"].shape[0]
        inference_input = {}
        for key in background.keys():
            inference_input[key] = torch.cat([background[key], target[key]], dim=0)
        outputs = self._generic_inference(**inference_input, n_samples=n_samples)
        batch_size_dim = 0 if n_samples == 1 else 1
        # split back vectors and assign to dictionaries
        background_outputs, target_outputs = {}, {}
        for key in outputs.keys():
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        # zero salient variables in background
        for key in ["s_exp", "s_chr", "s_pro", "s"]:
            background_outputs[key] = torch.zeros_like(background_outputs[key])
        return {"background": background_outputs, "target": target_outputs}

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: dict[str, dict[str, torch.Tensor]], index: str
    ) -> dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = {"batch_index": batch_index}
        return input_dict

    @staticmethod
    def _get_generative_input_from_inference_outputs(
        inference_outputs: dict[str, dict[str, torch.Tensor]], data_source: str
    ) -> dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library_rna = inference_outputs[data_source]["library_rna"]
        library_acc = inference_outputs[data_source]["library_acc"]
        return {"z": z, "s": s, "library_rna": library_rna, "library_acc": library_acc}

    def _get_generative_input(
        self,
        concat_tensors: dict[str, dict[str, torch.Tensor]],
        inference_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        background_tensor_input = self._get_generative_input_from_concat_tensors(concat_tensors, "background")
        target_tensor_input = self._get_generative_input_from_concat_tensors(concat_tensors, "target")
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensor_input, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensor_input, min_batch_size)

        background_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "background"
        )
        target_inference_outputs = self._get_generative_input_from_inference_outputs(inference_outputs, "target")
        background = {**background_tensor_input, **background_inference_outputs}
        target = {**target_tensor_input, **target_inference_outputs}
        return {"background": background, "target": target}

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library_rna: torch.Tensor,
        library_acc: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # create latent variables
        latent = torch.cat([z, s], dim=-1)

        # pass it through several decoders

        # gene expression
        px_scale, _, px_rate, _ = self.decoder_expression(
            "gene",
            latent,
            library_rna,
            batch_index,
        )
        # transform params if necessary
        px_r = torch.exp(self.px_r)

        # accessibility
        paccess_scale, _, paccess_rate, _ = self.decoder_accessibility("gene", latent, library_acc, batch_index)

        # Protein Decoder
        py_, log_pro_back_mean = self.decoder_pro(latent, batch_index)
        # Protein Dispersion
        py_r = torch.exp(self.py_r)
        py_["r"] = py_r

        # add params to dictionary
        return {
            "paccess_scale": paccess_scale,
            "paccess_rate": paccess_rate,
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "py_": py_,
            "log_pro_back_mean": log_pro_back_mean,
        }

    @auto_move_data
    def generative(
        self,
        background: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        # concatenate vectors for decoding
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]
        generative_input = {}
        for key in ["z", "s", "library_rna", "library_acc"]:
            generative_input[key] = torch.cat([background[key], target[key]], dim=batch_size_dim)
        generative_input["batch_index"] = torch.cat([background["batch_index"], target["batch_index"]], dim=0)
        outputs = self._generic_generative(**generative_input)
        # split back into parameters
        background_outputs, target_outputs = {}, {}
        # split what needs to be split in main dict
        for key in [
            "px_scale",
            "px_rate",
            "paccess_scale",
            "paccess_rate",
            "log_pro_back_mean",
        ]:
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        # the same for py_
        background_outputs["py_"] = {}
        target_outputs["py_"] = {}
        for key in [
            "back_alpha",
            "back_beta",
            "rate_back",
            "fore_scale",
            "rate_fore",
            "mixing",
            "scale",
        ]:
            background_tensor, target_tensor = torch.split(
                outputs["py_"][key],
                [background_batch_size, target_batch_size],
                dim=batch_size_dim,
            )
            background_outputs["py_"][key] = background_tensor
            target_outputs["py_"][key] = target_tensor
        # copy the rest
        background_outputs["px_r"] = outputs["px_r"]
        target_outputs["px_r"] = outputs["px_r"]
        background_outputs["py_"]["r"] = outputs["py_"]["r"]
        target_outputs["py_"]["r"] = outputs["py_"]["r"]

        return {"background": background_outputs, "target": target_outputs}

    def _generic_loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        use_chem_prior: bool = False,
        weight_obs: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        label = tensors[REGISTRY_KEYS.LABELS_KEY]

        # reconstruction loss from accessbility
        paccess_rate = generative_outputs["paccess_rate"]
        rl_accessibility = -Poisson(paccess_rate).log_prob(x_chr).sum(dim=-1)

        # reconstruction loss from rna
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        rl_expr = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x_rna).sum(dim=-1)

        # reconstruction loss from proteins
        py_ = generative_outputs["py_"]
        py_conditional = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        rl_protein = -py_conditional.log_prob(y).sum(dim=-1)

        # sum up the losses
        recon_loss = rl_accessibility + rl_protein + rl_expr

        # now put together the prior
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qs_m = inference_outputs["qs_m"]
        qs_v = inference_outputs["qs_v"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_s_m = torch.zeros_like(qs_m)
        prior_s_v = torch.ones_like(qs_v)

        # prune out entries according to mask
        if use_chem_prior:
            # extract chemical specific means
            prior_s_m = torch.index_select(self.label_specific_prior_mean, 0, label[:, 0].long())  # batch x latent dim

        kl_z = kl(
            Normal(qz_m, qz_v.sqrt()),
            Normal(prior_z_m, prior_z_v.sqrt()),
        ).sum(dim=-1)
        kl_s = kl(
            Normal(qs_m, qs_v.sqrt()),
            Normal(prior_s_m, prior_s_v.sqrt()),
        ).sum(dim=-1)

        return {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "kl_s": kl_s,
        }

    def loss(
        self,
        concat_tensors: dict[str, dict[str, torch.Tensor]],
        inference_outputs: dict[str, dict[str, torch.Tensor]],
        generative_outputs: dict[str, dict[str, torch.Tensor]],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """
        Compute loss terms for MoCAVI.

        Args:
        ----
            concat_tensors: Tuple of data mini-batch. The first element contains
                background data mini-batch. The second element contains target data
                mini-batch.
            inference_outputs: Dictionary of inference step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            generative_outputs: Dictionary of generative step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            kl_weight: Importance weight for KL divergence of background and salient
                latent variables, relative to KL divergence of library size.

        Returns
        -------
            An scvi.module.base.LossRecorder instance that records the following:
            loss: One-dimensional tensor for overall loss used for optimization.
            reconstruction_loss: Reconstruction loss with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_local: KL divergence term with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_global: One-dimensional tensor for global KL divergence term.
        """
        background_tensors = concat_tensors["background"]
        target_tensors = concat_tensors["target"]
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensors, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensors, min_batch_size)

        background_losses = self._generic_loss(
            background_tensors,
            inference_outputs["background"],
            generative_outputs["background"],
        )
        target_losses = self._generic_loss(
            target_tensors,
            inference_outputs["target"],
            generative_outputs["target"],
            use_chem_prior=True,
        )
        # rebalance the two dataloaders (smaller gets seen more), by weighting the likelihood to underepresent the smaller one (no chem)
        # the value of the weight is set in the training loop
        w = self.background_weight
        recon_loss = w * background_losses["recon_loss"] + (1 - w) * target_losses["recon_loss"]
        kl_z = w * background_losses["kl_z"] + (1 - w) * target_losses["kl_z"]
        kl_s = (1 - w) * target_losses["kl_s"]

        # wasserstein loss for disentanglement
        wasserstein_loss = (
            torch.norm(inference_outputs["background"]["qs_m"], dim=-1) ** 2
            + torch.norm(torch.sqrt(inference_outputs["background"]["qs_v"]), dim=-1) ** 2
        )

        loss = torch.mean(recon_loss + kl_weight * (kl_z + kl_s + self.wasserstein_reg * wasserstein_loss))
        kl_local = {
            "kl_divergence_z": kl_z,
            "kl_divergence_s": kl_s,
        }
        kl_global = torch.tensor(0.0)

        # LossOutput internally sums the reconst_loss, kl_local, and kl_global
        # terms before logging, so we do the same for the `wasserstein_loss` term.
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_local,
            kl_global=kl_global,
            extra_metrics={"wasserstein_loss_sum": torch.sum(wasserstein_loss)},
        )

    @torch.no_grad()
    def sample(self):
        raise NotImplementedError

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self):
        raise NotImplementedError
