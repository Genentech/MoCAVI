import numpy as np
import pandas as pd
import pytest
import torch

from mocavi import EBMoCAVIModel


def copy_module_state_dict(module) -> dict[str, torch.Tensor]:
    copy = {}
    for name, param in module.state_dict().items():
        copy[name] = param.detach().cpu().clone()
    return copy


@pytest.fixture
def mock_n_input(mock_ebmocavi_adata_manager):
    return mock_ebmocavi_adata_manager.adata.X.shape[1]


@pytest.fixture
def mock_emp_prior_dim(mock_ebmocavi_model):
    return mock_ebmocavi_model.adata_manager.summary_stats.n_emp_prior


@pytest.fixture
def mock_label_embedding_dict(mock_ebmocavi_model, mock_emp_prior_dim):
    # Use label values actually present in obs["labels"] so shrinkage methods can find them
    adata = mock_ebmocavi_model.adata
    label_names = list(adata.obs["labels"].unique())

    # Simple but deterministic embeddings: [0,0,...], [1,1,...], ...
    label_dict = {}
    for i, lab in enumerate(label_names):
        label_dict[lab] = np.full((mock_emp_prior_dim,), float(i))
    return label_dict


@pytest.fixture
def mock_n_batch(mock_ebmocavi_adata_manager):
    return len(mock_ebmocavi_adata_manager.adata.obs["batch"].unique())


@pytest.fixture()
def mock_ebmocavi_model(mock_ebmocavi_adata, mock_n_input, mock_n_batch):
    EBMoCAVIModel.setup_anndata(
        mock_ebmocavi_adata,
        layer="raw_counts",
        batch_key="batch",
        labels_key="labels",
        protein_expression_obsm_key="protein_expression",
        empirical_prior_key="embeddings",
    )
    return EBMoCAVIModel(
        mock_ebmocavi_adata,
        n_genes=mock_n_input - 5,
        n_regions=5,
        n_hidden=16,
        n_batch=mock_n_batch,
        n_background_latent=4,
        n_salient_latent=4,
        n_layers_encoder=2,
    )


class TestEBMoCAVI:
    def test_setup_anndata(self, mock_ebmocavi_adata):
        EBMoCAVIModel.setup_anndata(
            mock_ebmocavi_adata,
            layer="raw_counts",
            batch_key="batch",
            labels_key="labels",
            protein_expression_obsm_key="protein_expression",
            empirical_prior_key="embeddings",
        )

    def test_train(
        self,
        mock_ebmocavi_model,
        mock_ebmocavi_background_indices,
        mock_ebmocavi_target_indices,
    ):
        mock_ebmocavi_model.train(
            background_indices=mock_ebmocavi_background_indices,
            target_indices=mock_ebmocavi_target_indices,
            max_epochs=1,
            batch_size=400,  # Unequal final batches to test edge cases.
            plan_kwargs={
                "lr": 0.000001,
            },
        )

    def test_maximum_likelihood_training(self, mock_ebmocavi_model):
        module = mock_ebmocavi_model.module

        returned = mock_ebmocavi_model.maximum_likelihood_training(
            max_epochs=1,
            batch_size=16,
            learning_rate=1e-5,
            n_samples_per_cell=3,
        )

        # Returns self
        assert returned is mock_ebmocavi_model

        # All params should be back to trainable
        for name, p in module.named_parameters():
            assert p.requires_grad, f"{name} not unfrozen after MLE training"

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    def test_get_latent_representation(self, mock_ebmocavi_model, representation_kind):
        n_cells = mock_ebmocavi_model.adata.n_obs
        if representation_kind == "background":
            n_latent = mock_ebmocavi_model.module.n_background_latent
        else:
            n_latent = mock_ebmocavi_model.module.n_salient_latent
        representation = mock_ebmocavi_model.get_latent_representation(representation_kind=representation_kind)
        assert representation.shape == (n_cells, n_latent)

    @pytest.mark.parametrize("partial_modality", ["exp", "pro", "chr"])
    def test_get_latent_representation_partial(self, mock_ebmocavi_model, partial_modality):
        n_cells = mock_ebmocavi_model.adata.n_obs
        for representation_kind in ["background", "salient"]:
            if representation_kind == "background":
                n_latent = mock_ebmocavi_model.module.n_background_latent
            else:
                n_latent = mock_ebmocavi_model.module.n_salient_latent
        representation = mock_ebmocavi_model.get_latent_representation(
            representation_kind=representation_kind,
            partial_modality=partial_modality,
        )
        assert representation.shape == (n_cells, n_latent)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    def test_latent_importance(self, mock_ebmocavi_model, representation_kind):
        n_cells = mock_ebmocavi_model.adata.n_obs
        importance = mock_ebmocavi_model.get_latent_importance(representation_kind=representation_kind)
        assert importance["exp"].shape == (n_cells,)
        assert importance["chr"].shape == (n_cells,)
        assert importance["pro"].shape == (n_cells,)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_get_normalized_outputs(self, mock_ebmocavi_model, representation_kind, modality):
        n_samples = 50
        n_cells = mock_ebmocavi_model.adata.n_obs
        if modality == "exp":
            n_features = mock_ebmocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_ebmocavi_model.n_proteins
        else:
            n_features = mock_ebmocavi_model.n_regions

        one_sample_feats = mock_ebmocavi_model.get_normalized_outputs(n_samples=1)
        one_sample_feats = one_sample_feats[representation_kind][modality]
        assert isinstance(one_sample_feats, np.ndarray)

        assert one_sample_feats.shape == (n_cells, n_features)

        many_sample_exprs = mock_ebmocavi_model.get_normalized_outputs(n_samples=n_samples, return_mean=False)
        many_sample_exprs = many_sample_exprs[representation_kind][modality]
        assert isinstance(many_sample_exprs, np.ndarray)
        assert many_sample_exprs.shape == (n_samples, n_cells, n_features)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_get_specific_normalized_outputs(self, mock_ebmocavi_model, representation_kind, modality):
        n_samples = 50
        n_cells = mock_ebmocavi_model.adata.n_obs
        if modality == "exp":
            n_features = mock_ebmocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_ebmocavi_model.n_proteins
        else:
            n_features = mock_ebmocavi_model.n_regions
        one_sample_feats = mock_ebmocavi_model.get_specific_normalized_outputs(
            modality_key=modality, output_type=representation_kind, n_samples=1
        )
        assert isinstance(one_sample_feats, np.ndarray)

        assert one_sample_feats.shape == (n_cells, n_features)

        many_sample_exprs = mock_ebmocavi_model.get_specific_normalized_outputs(
            modality_key=modality, output_type=representation_kind, n_samples=n_samples, return_mean=False
        )
        assert isinstance(many_sample_exprs, np.ndarray)
        assert many_sample_exprs.shape == (n_samples, n_cells, n_features)

    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_differential_abundance(self, mock_ebmocavi_model, modality):
        if modality == "exp":
            n_features = mock_ebmocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_ebmocavi_model.n_proteins
        else:
            n_features = mock_ebmocavi_model.n_regions
        de_df = mock_ebmocavi_model.differential_abundance(
            groupby="labels",
            group1=["label_1"],
            modality_key=modality,
        )
        assert isinstance(de_df, pd.DataFrame)
        assert de_df.shape[0] == n_features

    @pytest.mark.parametrize("modality", ["exp", "pro", "chr"])
    def test_synthetic_differential_abundance(self, mock_ebmocavi_model, modality):
        if modality == "exp":
            n_features = mock_ebmocavi_model.n_genes
        elif modality == "pro":
            n_features = mock_ebmocavi_model.n_proteins
        else:
            n_features = mock_ebmocavi_model.n_regions
        de_df = mock_ebmocavi_model.synthetic_differential_abundance(
            groupby="labels",
            group1=["label_1"],
            modality_key=modality,
        )
        assert isinstance(de_df, pd.DataFrame)
        assert de_df.shape[0] == n_features

    def test_synthetic_fold_change(self, mock_ebmocavi_model):
        de_df = mock_ebmocavi_model.get_label_synthetic_fold_change()["fold_change_exp"]

        assert isinstance(de_df, pd.DataFrame)
        assert de_df.shape[1] == mock_ebmocavi_model.n_genes

    def test_get_label_total_fold_change(
        self,
        mock_ebmocavi_model,
    ):
        model = mock_ebmocavi_model
        n_labels = len(model.adata.obs["labels"].unique())
        n_genes = model.n_genes
        n_proteins = model.n_proteins
        n_regions = model.n_regions

        result = model.get_label_total_fold_change(
            batch_size=8,
            n_samples=5,
        )

        assert set(result.keys()) == {
            "fold_change_exp",
            "fold_change_pro",
            "fold_change_chr",
        }

        fc_exp = result["fold_change_exp"]
        fc_pro = result["fold_change_pro"]
        fc_chr = result["fold_change_chr"]

        assert isinstance(fc_exp, pd.DataFrame)
        assert isinstance(fc_pro, pd.DataFrame)
        assert isinstance(fc_chr, pd.DataFrame)

        assert fc_exp.shape == (n_labels, n_genes)
        assert fc_pro.shape == (n_labels, n_proteins)
        assert fc_chr.shape == (n_labels, n_regions)

    def test_get_label_shrinkage_total_fold_change(
        self,
        mock_ebmocavi_model,
        mock_label_embedding_dict,
    ):
        model = mock_ebmocavi_model
        n_labels = len(mock_label_embedding_dict)
        n_genes = model.n_genes
        n_proteins = model.n_proteins
        n_regions = model.n_regions

        result = model.get_label_shrinkage_total_fold_change(
            labels=mock_label_embedding_dict,
            adata=model.adata,
            batch_size=8,
            n_samples=5,
        )

        assert set(result.keys()) == {
            "fold_change_exp",
            "fold_change_pro",
            "fold_change_chr",
            "normalized_exp",
            "salient",
            "background",
        }

        fc_exp = result["fold_change_exp"]
        fc_pro = result["fold_change_pro"]
        fc_chr = result["fold_change_chr"]

        assert fc_exp.shape == (n_labels, n_genes)
        assert fc_pro.shape == (n_labels, n_proteins)
        assert fc_chr.shape == (n_labels, n_regions)

    def test_get_label_prior_total_fold_change(
        self,
        mock_ebmocavi_model,
        mock_label_embedding_dict,
    ):
        model = mock_ebmocavi_model
        n_labels = len(mock_label_embedding_dict)
        n_genes = model.n_genes
        n_proteins = model.n_proteins
        n_regions = model.n_regions

        result = model.get_label_prior_total_fold_change(
            labels=mock_label_embedding_dict,
            n_samples=5,
        )

        assert set(result.keys()) == {
            "fold_change_exp",
            "fold_change_pro",
            "fold_change_chr",
            "normalized_exp",
            "salient",
            "background",
        }

        assert result["fold_change_exp"].shape == (n_labels, n_genes)
        assert result["fold_change_pro"].shape == (n_labels, n_proteins)
        assert result["fold_change_chr"].shape == (n_labels, n_regions)

    def test_get_label_prior_synthetic_fold_change(
        self,
        mock_ebmocavi_model,
        mock_label_embedding_dict,
    ):
        model = mock_ebmocavi_model
        n_labels = len(mock_label_embedding_dict)
        n_genes = model.n_genes
        n_proteins = model.n_proteins
        n_regions = model.n_regions

        result = model.get_label_prior_synthetic_fold_change(
            labels=mock_label_embedding_dict,
            n_samples=5,
        )

        assert set(result.keys()) == {
            "fold_change_exp",
            "fold_change_pro",
            "fold_change_chr",
        }

        assert result["fold_change_exp"].shape == (n_labels, n_genes)
        assert result["fold_change_pro"].shape == (n_labels, n_proteins)
        assert result["fold_change_chr"].shape == (n_labels, n_regions)

    def test_get_label_prior_embedding_salient_and_background(
        self,
        mock_ebmocavi_model,
        mock_label_embedding_dict,
    ):
        model = mock_ebmocavi_model
        n_labels = len(mock_label_embedding_dict)

        salient = model.get_label_prior_embedding(
            labels=mock_label_embedding_dict,
            representation_kind="salient",
        )
        background = model.get_label_prior_embedding(
            labels=mock_label_embedding_dict,
            representation_kind="background",
        )

        assert isinstance(salient, pd.DataFrame)
        assert isinstance(background, pd.DataFrame)

        assert salient.shape[0] == n_labels
        assert background.shape[0] == n_labels

        assert salient.shape[1] == model.module.n_salient_latent
        assert background.shape[1] == model.module.n_background_latent
