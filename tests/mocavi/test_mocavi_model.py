import numpy as np
import pandas as pd
import pytest
import torch

from mocavi import MoCAVIModel


def copy_module_state_dict(module) -> dict[str, torch.Tensor]:
    copy = {}
    for name, param in module.state_dict().items():
        copy[name] = param.detach().cpu().clone()
    return copy


@pytest.fixture
def mock_n_input(mock_mocavi_adata_manager):
    return mock_mocavi_adata_manager.adata.X.shape[1]


@pytest.fixture
def mock_n_batch(mock_mocavi_adata_manager):
    return len(mock_mocavi_adata_manager.adata.obs["batch"].unique())


@pytest.fixture()
def mock_mocavi_model(mock_mocavi_adata, mock_n_input, mock_n_batch):
    MoCAVIModel.setup_anndata(
        mock_mocavi_adata,
        layer="raw_counts",
        batch_key="batch",
        labels_key="labels",
        protein_expression_obsm_key="protein_expression",
    )
    return MoCAVIModel(
        mock_mocavi_adata,
        n_genes=mock_n_input - 5,
        n_regions=5,
        n_hidden=16,
        n_batch=mock_n_batch,
        n_background_latent=4,
        n_salient_latent=4,
        n_layers_encoder=2,
    )


class TestMoCAVI:
    def test_setup_anndata(self, mock_mocavi_adata):
        MoCAVIModel.setup_anndata(
            mock_mocavi_adata,
            layer="raw_counts",
            batch_key="batch",
            labels_key="labels",
            protein_expression_obsm_key="protein_expression",
        )

    def test_train(
        self,
        mock_mocavi_model,
        mock_background_indices,
        mock_target_indices,
    ):
        mock_mocavi_model.train(
            background_indices=mock_background_indices,
            target_indices=mock_target_indices,
            max_epochs=1,
            batch_size=40,  # Unequal final batches to test edge cases.
            plan_kwargs={
                "lr": 0.0001,
            },
        )

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    def test_get_latent_representation(self, mock_mocavi_model, representation_kind):
        n_cells = mock_mocavi_model.adata.n_obs
        if representation_kind == "background":
            n_latent = mock_mocavi_model.module.n_background_latent
        else:
            n_latent = mock_mocavi_model.module.n_salient_latent
        representation = mock_mocavi_model.get_latent_representation(representation_kind=representation_kind)
        assert representation.shape == (n_cells, n_latent)

    @pytest.mark.parametrize("partial_modality", ["exp", "pro", "chr"])
    def test_get_latent_representation_partial(self, mock_mocavi_model, partial_modality):
        n_cells = mock_mocavi_model.adata.n_obs
        for representation_kind in ["background", "salient"]:
            if representation_kind == "background":
                n_latent = mock_mocavi_model.module.n_background_latent
            else:
                n_latent = mock_mocavi_model.module.n_salient_latent
        representation = mock_mocavi_model.get_latent_representation(
            representation_kind=representation_kind,
            partial_modality=partial_modality,
        )
        assert representation.shape == (n_cells, n_latent)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    def test_latent_importance(self, mock_mocavi_model, representation_kind):
        n_cells = mock_mocavi_model.adata.n_obs
        importance = mock_mocavi_model.get_latent_importance(representation_kind=representation_kind)
        assert importance["exp"].shape == (n_cells,)
        assert importance["chr"].shape == (n_cells,)
        assert importance["pro"].shape == (n_cells,)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_get_normalized_outputs(self, mock_mocavi_model, representation_kind, modality):
        n_samples = 50
        n_cells = mock_mocavi_model.adata.n_obs
        if modality == "exp":
            n_features = mock_mocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_mocavi_model.n_proteins
        else:
            n_features = mock_mocavi_model.n_regions

        one_sample_feats = mock_mocavi_model.get_normalized_outputs(n_samples=1)
        one_sample_feats = one_sample_feats[representation_kind][modality]
        assert isinstance(one_sample_feats, np.ndarray)

        assert one_sample_feats.shape == (n_cells, n_features)

        many_sample_exprs = mock_mocavi_model.get_normalized_outputs(n_samples=n_samples, return_mean=False)
        many_sample_exprs = many_sample_exprs[representation_kind][modality]
        assert isinstance(many_sample_exprs, np.ndarray)
        assert many_sample_exprs.shape == (n_samples, n_cells, n_features)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_get_specific_normalized_outputs(self, mock_mocavi_model, representation_kind, modality):
        n_samples = 50
        n_cells = mock_mocavi_model.adata.n_obs
        if modality == "exp":
            n_features = mock_mocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_mocavi_model.n_proteins
        else:
            n_features = mock_mocavi_model.n_regions

        one_sample_feats = mock_mocavi_model.get_specific_normalized_outputs(
            modality_key=modality, output_type=representation_kind, n_samples=1
        )
        assert isinstance(one_sample_feats, np.ndarray)

        assert one_sample_feats.shape == (n_cells, n_features)

        many_sample_exprs = mock_mocavi_model.get_specific_normalized_outputs(
            modality_key=modality, output_type=representation_kind, n_samples=n_samples, return_mean=False
        )
        assert isinstance(many_sample_exprs, np.ndarray)
        assert many_sample_exprs.shape == (n_samples, n_cells, n_features)

    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_differential_abundance(self, mock_mocavi_model, modality):
        if modality == "exp":
            n_features = mock_mocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_mocavi_model.n_proteins
        else:
            n_features = mock_mocavi_model.n_regions
        de_df = mock_mocavi_model.differential_abundance(
            groupby="labels",
            group1=["label_0"],
            modality_key=modality,
        )
        assert isinstance(de_df, pd.DataFrame)
        assert de_df.shape[0] == n_features

    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_synthetic_differential_abundance(self, mock_mocavi_model, modality):
        if modality == "exp":
            n_features = mock_mocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_mocavi_model.n_proteins
        else:
            n_features = mock_mocavi_model.n_regions
        de_df = mock_mocavi_model.synthetic_differential_abundance(
            groupby="labels",
            group1=["label_0"],
            modality_key=modality,
        )
        assert isinstance(de_df, pd.DataFrame)
        assert de_df.shape[0] == n_features

    def test_synthetic_fold_change(self, mock_mocavi_model):
        de_df = mock_mocavi_model.get_label_synthetic_fold_change()["fold_change_exp"]

        assert isinstance(de_df, pd.DataFrame)
        assert de_df.shape[1] == mock_mocavi_model.n_genes
