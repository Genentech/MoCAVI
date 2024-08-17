"""Model class for MoCAVI for single cell multi-omics data."""

import logging
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._types import Number
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    ProteinObsmField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import (
    _get_batch_code_from_category,
    scrna_raw_counts_properties,
)
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from ._differential import _de_core, _de_paired_test_core, adt_raw_counts_properties
from ._module import MoCAVIModule
from ._training_mixin import MoCAVITrainingMixin

logger = logging.getLogger(__name__)


class MoCAVIModel(MoCAVITrainingMixin, BaseModelClass):
    """
    Model class for MoCAVI.

    Args:
    ----
        adata: AnnData object that has been registered via
            `MoCAVIModel.setup_anndata`.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.

    """

    def __init__(
        self,
        adata: AnnData,
        n_genes: int,
        n_regions: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate: float = 0.1,
        reweighting_mode: str = "bt",
        wasserstein_reg: float = 0,
    ) -> None:
        super().__init__(adata)
        self.reweighting_mode = reweighting_mode

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        # overload summary stats
        self.n_genes = n_genes
        self.n_regions = n_regions
        n_proteins = self.summary_stats.get("n_proteins", 0)
        self.n_proteins = n_proteins
        n_batch = self.summary_stats.n_batch

        # create module
        self.module = MoCAVIModule(
            n_input_genes=n_genes,
            n_input_regions=n_regions,
            n_input_proteins=n_proteins,
            n_labels=self.summary_stats["n_labels"],
            n_cats_per_cov=n_cats_per_cov,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            dropout_rate=dropout_rate,
            wasserstein_reg=wasserstein_reg,
        )

        self._model_summary_string = "MoCAVI Model with the following params:\n"
        self._model_summary_string += f"n_genes: {n_genes}\n"
        self._model_summary_string += f"n_regions: {n_regions}\n"
        self._model_summary_string += f"n_proteins: {n_proteins}\n"
        self._model_summary_string += f"n_batch: {n_batch}\n"
        self._model_summary_string += f"n_hidden: {n_hidden}\n"
        self._model_summary_string += f"n_background_latent: {n_background_latent}\n"
        self._model_summary_string += f"n_salient_latent: {n_salient_latent}\n"
        self._model_summary_string += f"n_layers_encoder: {n_layers_encoder}\n"
        self._model_summary_string += f"n_layers_decoder: {n_layers_decoder}\n"
        self._model_summary_string += f"dropout_rate: {dropout_rate}\n"
        self._model_summary_string += f"reweighting_mode: {reweighting_mode}\n"

        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        protein_names_uns_key: str | None = None,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ) -> AnnData | None:
        """
        Set up AnnData instance for MoCAVIVI model.

        Args:
        ----
            adata: AnnData object containing raw counts. Rows represent cells, columns
                represent features.
            layer: If not None, uses this as the key in adata.layers for raw count data.
            batch_key: Key in `adata.obs` for batch information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_batch"]`. If None, assign the same batch to all the
                data.
            labels_key: Key in `adata.obs` for label information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_labels"]`. If None, assign the same label to all the
                data.
            size_factor_key: Key in `adata.obs` for size factor information. Instead of
                using library size as a size factor, the provided size factor column
                will be used as offset in the mean of the likelihood. Assumed to be on
                linear scale.
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Used in some models.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Used in some models.

        Returns
        -------
            If `copy` is True, return the modified `adata` set up for MoCAVI
            model, otherwise `adata` is modified in place.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            batch_field,
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_importance(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """
        Return the importance of each modality for either background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of" f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent_imp_exp = []
        latent_imp_pro = []
        latent_imp_chr = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            outputs = self.module._generic_inference(x=x, y=y, batch_index=batch_index, n_samples=1)

            key = "z" if representation_kind == "background" else "s"
            mus = torch.stack(
                [
                    outputs[f"q{key}_exp_m"],
                    outputs[f"q{key}_chr_m"],
                    outputs[f"q{key}_pro_m"],
                ],
                dim=-1,
            )  # batch, dimension, modality
            vars = torch.stack(
                [
                    outputs[f"q{key}_exp_v"],
                    outputs[f"q{key}_chr_v"],
                    outputs[f"q{key}_pro_v"],
                ],
                dim=-1,
            )  # batch, dimension, modality
            mus_joint = torch.sum(mus / vars, dim=-1)  # batch size, dimension
            vars_joint = torch.zeros_like(mus_joint) + torch.sum(1.0 / vars, dim=-1)
            vars_joint = 1.0 / vars_joint  # inverse
            mus_joint *= vars_joint
            top_vector = vars_joint[:, :, None] * mus / (vars)
            mus_contrib = torch.sum((top_vector) ** 2, dim=1) / torch.sum(
                mus_joint[:, :, None] ** 2, dim=1
            )  # batch, modality
            latent_imp_exp += [mus_contrib[:, 0].detach().cpu()]
            latent_imp_chr += [mus_contrib[:, 1].detach().cpu()]
            latent_imp_pro += [mus_contrib[:, 2].detach().cpu()]
        return {
            "exp": torch.cat(latent_imp_exp).numpy(),
            "chr": torch.cat(latent_imp_chr).numpy(),
            "pro": torch.cat(latent_imp_pro).numpy(),
        }

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        representation_kind: str = "salient",
        partial_modality: str | None = None,
    ) -> np.ndarray:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of" f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            outputs = self.module._generic_inference(x=x, y=y, batch_index=batch_index, n_samples=1)

            if partial_modality is not None and partial_modality != "":
                give_mean = True
                if partial_modality == "exp":
                    key = "z_exp" if representation_kind == "background" else "s_exp"
                elif partial_modality == "pro":
                    key = "z_pro" if representation_kind == "background" else "s_pro"
                elif partial_modality == "chr":
                    key = "z_chr" if representation_kind == "background" else "s_chr"
                else:
                    raise ValueError(f"partial_modality = {partial_modality} is not one of" f" ['exp', 'pro', 'chr']")
                latent_sample = outputs[f"q{key}_m"]
            else:
                # full modality
                if representation_kind == "background":
                    latent_m = outputs["qz_m"]
                    latent_sample = outputs["z"]
                else:
                    latent_m = outputs["qs_m"]
                    latent_sample = outputs["s"]

                if give_mean:
                    latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.inference_mode()
    def get_normalized_outputs(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        disable_tqdm: bool | None = False,
    ) -> dict[str, np.ndarray | pd.DataFrame]:
        """
        Return the normalized (decoded) outputs.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.

        Returns
        -------
            A dictionary with keys "background" and "salient", with value as follows.
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = _get_batch_code_from_category(self.get_anndata_manager(adata, required=True), transform_batch)

        background_exprs = []
        salient_exprs = []
        background_pros = []
        salient_pros = []
        background_chrs = []
        salient_chrs = []
        for tensors in tqdm(data_loader, disable=disable_tqdm):
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            background_per_batch_exprs = []
            salient_per_batch_exprs = []
            background_per_batch_pros = []
            salient_per_batch_pros = []
            background_per_batch_chrs = []
            salient_per_batch_chrs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_index = torch.ones_like(batch_index) * batch

                # first get latent variables
                inference_outputs = self.module._generic_inference(
                    x=x, y=y, batch_index=batch_index, n_samples=n_samples
                )
                z = inference_outputs["z"]
                s = inference_outputs["s"]
                library_rna = inference_outputs["library_rna"]
                library_acc = inference_outputs["library_acc"]

                # second, get imputed values for background
                background_generative_outputs = self.module._generic_generative(
                    z=z,
                    s=torch.zeros_like(s),
                    library_rna=library_rna,
                    library_acc=library_acc,
                    batch_index=batch_index,
                )
                background_outputs = self._scale_and_numpy(background_generative_outputs)
                (
                    background_exprs_,
                    background_pros_,
                    background_chrs_,
                ) = background_outputs
                background_per_batch_exprs.append(background_exprs_)
                background_per_batch_pros.append(background_pros_)
                background_per_batch_chrs.append(background_chrs_)

                # now get them for salient
                salient_generative_outputs = self.module._generic_generative(
                    z=z,
                    s=s,
                    library_rna=library_rna,
                    library_acc=library_acc,
                    batch_index=batch_index,
                )
                salient_outputs = self._scale_and_numpy(salient_generative_outputs)
                salient_exprs_, salient_pros_, salient_chrs_ = salient_outputs
                salient_per_batch_exprs.append(salient_exprs_)
                salient_per_batch_pros.append(salient_pros_)
                salient_per_batch_chrs.append(salient_chrs_)

            # then format and assemble values,
            # Shape above is (len(transform_batch) x batch_size x n_var).
            background_per_batch_exprs = np.stack(background_per_batch_exprs)
            salient_per_batch_exprs = np.stack(salient_per_batch_exprs)
            background_per_batch_pros = np.stack(background_per_batch_pros)
            salient_per_batch_pros = np.stack(salient_per_batch_pros)
            background_per_batch_chrs = np.stack(background_per_batch_chrs)
            salient_per_batch_chrs = np.stack(salient_per_batch_chrs)
            # average per batch (note: it might be better to average the LFC, not the GEX...)
            background_exprs += [background_per_batch_exprs.mean(0)]
            salient_exprs += [salient_per_batch_exprs.mean(0)]
            background_pros += [background_per_batch_pros.mean(0)]
            salient_pros += [salient_per_batch_pros.mean(0)]
            background_chrs += [background_per_batch_chrs.mean(0)]
            salient_chrs += [salient_per_batch_chrs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            background_exprs = np.concatenate(background_exprs, axis=-2)
            salient_exprs = np.concatenate(salient_exprs, axis=-2)
            background_pros = np.concatenate(background_pros, axis=-2)
            salient_pros = np.concatenate(salient_pros, axis=-2)
            background_chrs = np.concatenate(background_chrs, axis=-2)
            salient_chrs = np.concatenate(salient_chrs, axis=-2)
        else:
            background_exprs = np.concatenate(background_exprs, axis=0)
            salient_exprs = np.concatenate(salient_exprs, axis=0)
            background_pros = np.concatenate(background_pros, axis=0)
            salient_pros = np.concatenate(salient_pros, axis=0)
            background_chrs = np.concatenate(background_chrs, axis=0)
            salient_chrs = np.concatenate(salient_chrs, axis=0)

        if n_samples > 1 and return_mean:
            background_exprs = background_exprs.mean(0)
            salient_exprs = salient_exprs.mean(0)
            background_pros = background_pros.mean(0)
            salient_pros = salient_pros.mean(0)
            background_chrs = background_chrs.mean(0)
            salient_chrs = salient_chrs.mean(0)

        return {
            "background": {
                "exp": background_exprs,
                "pro": background_pros,
                "chr": background_chrs,
            },
            "salient": {"exp": salient_exprs, "pro": salient_pros, "chr": salient_chrs},
        }

    @torch.inference_mode()
    def get_specific_normalized_outputs(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        output_type: str | None = None,
        indices_to_return_salient: Sequence[int] | None = None,
        modality_key: str | None = "exp",  # one of ["exp", "pro", "chr"]
        disable_tqdm: bool | None = False,
    ):
        """
        Return the normalized (decoded) gene output.

        Outputs are decoded from either the background or salient latent space.
        One of `output_type` or `indices_to_return_salient` should have an input
        argument.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        output_type: One of {"salient", "background"} to specify the type of
            normalized expression to return.
        indices_to_return_salient: If `indices` is a subset of
            `indices_to_return_salient`, normalized expressions derived from background
            and salient latent embeddings are returned. If `indices` is not `None` and
            is not a subset of `indices_to_return_salient`, normalized expressions
            derived only from background latent embeddings are returned.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`.
        """
        is_output_type_none = output_type is None
        is_indices_to_return_salient_none = indices_to_return_salient is None
        if is_output_type_none and is_indices_to_return_salient_none:
            raise ValueError(
                "Both expression_type and indices_to_return_salient are None! "
                "Exactly one of them needs to be supplied with an input argument."
            )
        elif (not is_output_type_none) and (not is_indices_to_return_salient_none):
            raise ValueError(
                "Both expression_type and indices_to_return_salient have an input "
                "argument! Exactly one of them needs to be supplied with an input "
                "argument."
            )
        else:
            outputs = self.get_normalized_outputs(
                adata=adata,
                indices=indices,
                transform_batch=transform_batch,
                n_samples=n_samples,
                n_samples_overall=n_samples_overall,
                batch_size=batch_size,
                return_mean=return_mean,
                disable_tqdm=disable_tqdm,
            )
            if not is_output_type_none:
                return outputs[output_type][modality_key]
            else:
                if indices is None:
                    indices = np.arange(adata.n_obs)
                if set(indices).issubset(set(indices_to_return_salient)):
                    return outputs["salient"][modality_key]
                else:
                    return outputs["background"][modality_key]

    def differential_abundance(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | str | None = None,
        idx2: Sequence[int] | Sequence[bool] | str | None = None,
        modality_key: str | None = "exp",  # one of ["exp", "pro", "chr"]
        mode: str = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        target_idx: Sequence[int] | None = None,
        n_samples: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Perform differential expression analysis.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        groupby: The key of the observations grouping to consider.
        group1: Subset of groups, e.g. ["g1", "g2", "g3"], to which comparison shall be
            restricted, or all groups in `groupby` (default).
        group2: If `None`, compare each group in `group1` to the union of the rest of
            the groups in `groupby`. If a group identifier, compare with respect to this
            group.
        idx1: `idx1` and `idx2` can be used as an alternative to the AnnData keys.
            Custom identifier for `group1` that can be of three sorts:
            (1) a boolean mask, (2) indices, or (3) a string. If it is a string, then
            it will query indices that verifies conditions on adata.obs, as described
            in `pandas.DataFrame.query()`. If `idx1` is not `None`, this option
            overrides `group1` and `group2`.
        idx2: Custom identifier for `group2` that has the same properties as `idx1`.
            By default, includes all cells not specified in `idx1`.
        mode: Method for differential expression. See
            https://docs.scvi-tools.org/en/0.14.1/user_guide/background/differential_expression.html
            for more details.
        delta: Specific case of region inducing differential expression. In this case,
            we suppose that R\[-delta, delta] does not induce differential expression
            (change model default case).
        batch_size: Mini-batch size for data loading into model. Defaults to
            scvi.settings.batch_size.
        all_stats: Concatenate count statistics (e.g., mean expression group 1) to DE
            results.
        batch_correction: Whether to correct for batch effects in DE inference.
        batchid1: Subset of categories from `batch_key` registered in `setup_anndata`,
            e.g. ["batch1", "batch2", "batch3"], for `group1`. Only used if
            `batch_correction` is `True`, and by default all categories are used.
        batchid2: Same as `batchid1` for `group2`. `batchid2` must either have null
            intersection with `batchid1`, or be exactly equal to `batchid1`. When the
            two sets are exactly equal, cells are compared by decoding on the same
            batch. When sets have null intersection, cells from `group1` and `group2`
            are decoded on each group in `group1` and `group2`, respectively.
        fdr_target: Tag features as DE based on posterior expected false discovery rate.
        silent: If `True`, disables the progress bar. Default: `False`.
        target_idx: If not `None`, a boolean or integer identifier should be used for
            cells in the target group. Normalized expression values derived
            from both salient and background latent embeddings are used when
            {group1, group2} is a subset of the target group, otherwise background
            normalized expression values are used.

        **kwargs: Keyword args for
            `scvi.model.base.DifferentialComputation.get_bayes_factors`.

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        if target_idx is not None:
            target_idx = np.array(target_idx)
            if target_idx.dtype is np.dtype("bool"):
                assert len(target_idx) == adata.n_obs, "target_idx mask must be the same length as adata!"
                target_idx = np.arange(adata.n_obs)[target_idx]
            model_fn = partial(
                self.get_specific_normalized_outputs,
                n_samples=n_samples,
                batch_size=batch_size,
                output_type=None,
                modality_key=modality_key,
                indices_to_return_salient=target_idx,
                disable_tqdm=True,
            )
        else:
            model_fn = partial(
                self.get_specific_normalized_outputs,
                n_samples=n_samples,
                batch_size=batch_size,
                output_type="salient",
                modality_key=modality_key,
                indices_to_return_salient=None,
                disable_tqdm=True,
            )

        if modality_key == "exp" or modality_key == "chr":
            var_idx = slice(0, self.n_genes) if modality_key == "exp" else slice(self.n_genes, None)
            col_names = adata.var_names[var_idx]
            raw_data_fn = partial(scrna_raw_counts_properties, var_idx=var_idx)
        else:  # pro
            var_idx = slice(None, None)
            col_names = adata.uns["protein_names"][var_idx]
            raw_data_fn = adt_raw_counts_properties

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=raw_data_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return result

    def synthetic_differential_abundance(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        idx1: Sequence[int] | Sequence[bool] | str | None = None,
        modality_key: str | None = "exp",  # one of ["exp", "pro", "chr"]
        mode: str = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        n_samples: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Perform differential expression analysis.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        groupby: The key of the observations grouping to consider.
        group1: Subset of groups, e.g. ["g1", "g2", "g3"], to which comparison shall be
            restricted, or all groups in `groupby` (default).
        idx1: `idx1` and `idx2` can be used as an alternative to the AnnData keys.
            Custom identifier for `group1` that can be of three sorts:
            (1) a boolean mask, (2) indices, or (3) a string. If it is a string, then
            it will query indices that verifies conditions on adata.obs, as described
            in `pandas.DataFrame.query()`. If `idx1` is not `None`, this option
            overrides `group1` and `group2`.
        mode: Method for differential expression. See
            https://docs.scvi-tools.org/en/0.14.1/user_guide/background/differential_expression.html
            for more details.
        delta: Specific case of region inducing differential expression. In this case,
            we suppose that R\[-delta, delta] does not induce differential expression
            (change model default case).
        batch_size: Mini-batch size for data loading into model. Defaults to
            scvi.settings.batch_size.
        all_stats: Concatenate count statistics (e.g., mean expression group 1) to DE
            results.
        batch_correction: Whether to correct for batch effects in DE inference.
        batchid1: Subset of categories from `batch_key` registered in `setup_anndata`,
            e.g. ["batch1", "batch2", "batch3"], for `group1`. Only used if
            `batch_correction` is `True`, and by default all categories are used.
        fdr_target: Tag features as DE based on posterior expected false discovery rate.
        silent: If `True`, disables the progress bar. Default: `False`.

        **kwargs: Keyword args for
            `scvi.model.base.DifferentialComputation.get_bayes_factors`.

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        model_fn1 = partial(
            self.get_specific_normalized_outputs,
            n_samples=n_samples,
            batch_size=batch_size,
            output_type="salient",
            modality_key=modality_key,
            indices_to_return_salient=None,
            disable_tqdm=True,
        )
        model_fn2 = partial(
            self.get_specific_normalized_outputs,
            n_samples=n_samples,
            batch_size=batch_size,
            output_type="background",
            modality_key=modality_key,
            indices_to_return_salient=None,
            disable_tqdm=True,
        )

        if modality_key == "exp" or modality_key == "chr":
            var_idx = slice(0, self.n_genes) if modality_key == "exp" else slice(self.n_genes, None)
            col_names = adata.var_names[var_idx]
            raw_data_fn = partial(scrna_raw_counts_properties, var_idx=var_idx)
        else:  # pro
            var_idx = slice(None, None)
            col_names = adata.uns["protein_names"][var_idx]
            raw_data_fn = adt_raw_counts_properties

        result = _de_paired_test_core(
            self.get_anndata_manager(adata, required=True),
            model_fn1,
            model_fn2,
            groupby=groupby,
            group1=group1,
            idx1=idx1,
            all_stats=all_stats,
            all_stats_fn=raw_data_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return result

    @staticmethod
    @torch.inference_mode()
    def _scale_and_numpy(
        generative_outputs: dict[str, torch.Tensor],
    ) -> np.ndarray:
        # expression
        output_expr = generative_outputs["px_scale"]
        output_expr = output_expr.cpu().numpy()
        # proteins
        py_ = generative_outputs["py_"]
        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        protein_val = py_["rate_fore"] * (1 - protein_mixing)
        output_pro = protein_val.cpu().numpy()
        # chromatin
        output_chr = generative_outputs["paccess_scale"]
        output_chr = output_chr.cpu().numpy()
        return output_expr, output_pro, output_chr

    @torch.inference_mode()
    def get_label_synthetic_fold_change(
        self,
        labels: Sequence[str] | None = None,
        batch_size: int | None = None,
        n_samples: int = 100,
        library_rna: float = 1e4,
        **kwargs,
    ):
        self._validate_anndata(self.adata)

        if labels is None:
            # get label naming conventions
            ordered_label_treatment_list = self.adata_manager.get_state_registry("labels").items()[0][1]
            indices = None
            n_labels = len(ordered_label_treatment_list)
        else:
            # check if all samples in adata have a label
            assert self.adata.n_obs == len(labels)
            # get label naming conventions
            ordered_label_treatment_list = np.unique(labels)
            indices = None
            n_labels = len(ordered_label_treatment_list)

        data_loader = self._make_data_loader(
            adata=self.adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        label_counts = np.zeros(shape=(n_labels,))
        salient_sum_exprs = np.zeros(shape=(n_labels, self.n_genes))
        salient_sum_pros = np.zeros(shape=(n_labels, self.summary_stats["n_proteins"]))
        salient_sum_chr = np.zeros(shape=(n_labels, self.n_regions))

        background_sum_exprs = np.zeros(shape=(n_labels, self.n_genes))
        background_sum_pros = np.zeros(shape=(n_labels, self.summary_stats["n_proteins"]))
        background_sum_chr = np.zeros(shape=(n_labels, self.n_regions))

        # loop through data, and get stochastic approximation of the sum of expression
        for t, tensors in tqdm(enumerate(data_loader)):
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            if labels is None:
                label = tensors[REGISTRY_KEYS.LABELS_KEY]
                label = label[:, 0].cpu().numpy().astype(int)

            else:
                label = labels[t * batch_size : (t + 1) * batch_size]
                label = np.array([np.where(ordered_label_treatment_list == x)[0][0] for x in label])

            # first get latent variables
            inference_outputs = self.module._generic_inference(x=x, y=y, batch_index=batch_index, n_samples=n_samples)
            z = inference_outputs["z"]
            s = inference_outputs["s"]
            library_rna_ = inference_outputs["library_rna"]
            library_acc_ = inference_outputs["library_acc"]

            # second, get imputed values from background
            background_generative_outputs = self.module._generic_generative(
                z=z,
                s=torch.zeros_like(s),
                library_rna=library_rna_,
                library_acc=library_acc_,
                batch_index=batch_index,
            )
            background_exprs_ = background_generative_outputs["px_scale"].mean(0).cpu().numpy()
            background_chrs_ = background_generative_outputs["paccess_scale"].mean(0).cpu().numpy()
            py_ = background_generative_outputs["py_"]
            background_pros_ = (py_["rate_fore"] * (1 - (1 / (1 + torch.exp(-py_["mixing"]))))).mean(0).cpu().numpy()

            # third, get imputed values from salient
            salient_generative_outputs = self.module._generic_generative(
                z=z,
                s=s,
                library_rna=library_rna_,
                library_acc=library_acc_,
                batch_index=batch_index,
            )
            salient_exprs_ = salient_generative_outputs["px_scale"].mean(0).cpu().numpy()
            salient_chrs_ = salient_generative_outputs["paccess_scale"].mean(0).cpu().numpy()
            py_ = salient_generative_outputs["py_"]
            salient_pros_ = (py_["rate_fore"] * (1 - (1 / (1 + torch.exp(-py_["mixing"]))))).mean(0).cpu().numpy()
            # fourth, inject imputed values into the sum of expression
            for i in np.unique(label):
                salient_sum_exprs[i] += salient_exprs_[label == i].sum(0)
                salient_sum_pros[i] += salient_pros_[label == i].sum(0)
                salient_sum_chr[i] += salient_chrs_[label == i].sum(0)

                background_sum_exprs[i] += background_exprs_[label == i].sum(0)
                background_sum_pros[i] += background_pros_[label == i].sum(0)
                background_sum_chr[i] += background_chrs_[label == i].sum(0)
                label_counts[i] += (label == i).sum()

        # Then, normalize
        salient_mean_exprs = salient_sum_exprs / label_counts[:, None]
        salient_mean_pros = salient_sum_pros / label_counts[:, None]
        salient_mean_chr = salient_sum_chr / label_counts[:, None]

        background_mean_exprs = background_sum_exprs / label_counts[:, None]
        background_mean_pros = background_sum_pros / label_counts[:, None]
        background_mean_chr = background_sum_chr / label_counts[:, None]

        # finally, get fold change
        fold_change_exprs = np.log2(1 + library_rna * salient_mean_exprs) - np.log2(
            1 + library_rna * background_mean_exprs
        )
        fold_change_pros = np.log2(0.5 + salient_mean_pros) - np.log2(0.5 + background_mean_pros)
        fold_change_chr = np.log2(1 + library_rna * salient_mean_chr) - np.log2(1 + library_rna * background_mean_chr)

        # put in a dataframe
        fold_change_exprs = pd.DataFrame(
            fold_change_exprs,
            index=ordered_label_treatment_list,
            columns=self.adata.var.index[: self.n_genes],
        )
        fold_change_chr = pd.DataFrame(
            fold_change_chr,
            index=ordered_label_treatment_list,
            columns=self.adata.var.index[self.n_genes :],
        )
        fold_change_pros = pd.DataFrame(
            fold_change_pros,
            index=ordered_label_treatment_list,
            columns=self.adata_manager.get_state_registry("proteins").values()[0],
        )

        # return in a dictionary
        return {
            "fold_change_exprs": fold_change_exprs,
            "fold_change_pros": fold_change_pros,
            "fold_change_chr": fold_change_chr,
        }
