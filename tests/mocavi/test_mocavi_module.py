import numpy as np
import pytest
import torch
from scvi.module.base import LossOutput

from mocavi._dataloader import MoCAVIDataLoader
from mocavi._module import MoCAVIModule

REQUIRED_DATA_SOURCES = ["background", "target"]
REQUIRED_INFERENCE_INPUT_KEYS = ["x", "y", "batch_index"]
REQUIRED_INFERENCE_OUTPUT_KEYS = [
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
REQUIRED_GENERATIVE_INPUT_KEYS_FROM_CONCAT_TENSORS = ["batch_index"]
REQUIRED_GENERATIVE_INPUT_KEYS_FROM_INFERENCE_OUTPUTS = ["z", "s", "library_rna", "library_acc"]
REQUIRED_GENERATIVE_OUTPUT_KEYS = [
    "px_scale",
    "px_r",
    "px_rate",
    "paccess_scale",
    "paccess_rate",
    "py_",
    "log_pro_back_mean",
]


@pytest.fixture
def mock_n_input(mock_mocavi_adata_manager):
    return mock_mocavi_adata_manager.adata.X.shape[1]


@pytest.fixture
def mock_n_labels(mock_mocavi_adata_manager):
    return len(np.unique(mock_mocavi_adata_manager.adata.obs["labels"]))


@pytest.fixture
def mock_n_batch(mock_mocavi_adata_manager):
    return len(mock_mocavi_adata_manager.adata.obs["batch"].unique())


@pytest.fixture
def mock_mocavi_dataloader(mock_mocavi_adata_manager, mock_background_indices, mock_target_indices):
    return MoCAVIDataLoader(
        mock_mocavi_adata_manager,
        mock_background_indices,
        mock_target_indices,
        batch_size=32,
        shuffle=False,
    )


@pytest.fixture
def mock_mocavi_batch(mock_mocavi_dataloader):
    batch = next(batch for batch in mock_mocavi_dataloader)
    return batch


@pytest.fixture
def mock_background_batch(mock_mocavi_adata_manager, mock_background_indices):
    background_batch = mock_mocavi_adata_manager.adata.obs["_scvi_batch"][mock_background_indices].unique()
    assert len(background_batch) == 1
    return background_batch.item()


@pytest.fixture
def mock_target_batch(mock_mocavi_adata_manager, mock_target_indices):
    target_batch = mock_mocavi_adata_manager.adata.obs["_scvi_batch"][mock_target_indices].unique()
    assert len(target_batch) == 1
    return target_batch.item()


@pytest.fixture
def mock_mocavi_module(mock_n_input, mock_n_batch, mock_n_labels):
    return MoCAVIModule(
        n_input_genes=mock_n_input,
        n_input_regions=0,
        n_input_proteins=0,
        n_labels=mock_n_labels,
        n_batch=mock_n_batch,
        n_hidden=10,
        n_background_latent=4,
        n_salient_latent=4,
        n_layers_decoder=2,
        n_layers_encoder=2,
    )


@pytest.fixture(params=[1, 2], ids=["one_latent_sample", "two_latent_samples"])
def mock_mocavi_vi_data(
    mock_mocavi_batch,
    mock_mocavi_module,
    request,
):
    concat_tensors = mock_mocavi_batch
    inference_input = mock_mocavi_module._get_inference_input(concat_tensors)
    inference_outputs = mock_mocavi_module.inference(**inference_input, n_samples=request.param)
    generative_input = mock_mocavi_module._get_generative_input(concat_tensors, inference_outputs)
    generative_outputs = mock_mocavi_module.generative(**generative_input)
    return {
        "concat_tensors": concat_tensors,
        "inference_input": inference_input,
        "inference_outputs": inference_outputs,
        "generative_input": generative_input,
        "generative_outputs": generative_outputs,
    }


class TestMoCAVIInference:
    def test_get_inference_input_from_concat_tensors(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
        mock_n_input,
    ):
        inference_input = mock_mocavi_module._get_inference_input_from_concat_tensors(mock_mocavi_batch, "background")
        for key in REQUIRED_INFERENCE_INPUT_KEYS:
            assert key in inference_input.keys()
        x = inference_input["x"]
        batch_index = inference_input["batch_index"]
        batch_size = x.shape[0]
        assert x.shape == (batch_size, mock_n_input)
        assert batch_index.shape == (batch_size, 1)

    def test_get_inference_input(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
        mock_background_batch,
        mock_target_batch,
    ):
        inference_input = mock_mocavi_module._get_inference_input(mock_mocavi_batch)
        for data_source in REQUIRED_DATA_SOURCES:
            assert data_source in inference_input.keys()

        background_input = inference_input["background"]
        background_input_keys = background_input.keys()
        target_input = inference_input["target"]
        target_input_keys = target_input.keys()

        for key in REQUIRED_INFERENCE_INPUT_KEYS:
            assert key in background_input_keys
            assert key in target_input_keys

        # Check background vs. target labels are consistent.
        assert (background_input["batch_index"] != mock_background_batch).sum() == 0
        assert (target_input["batch_index"] != mock_target_batch).sum() == 0

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_generic_inference(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
        n_samples,
    ):
        inference_input = mock_mocavi_module._get_inference_input_from_concat_tensors(mock_mocavi_batch, "background")
        batch_size = inference_input["x"].shape[0]
        n_background_latent = mock_mocavi_module.n_background_latent
        n_salient_latent = mock_mocavi_module.n_salient_latent

        inference_outputs = mock_mocavi_module._generic_inference(**inference_input, n_samples=n_samples)
        for key in REQUIRED_INFERENCE_OUTPUT_KEYS:
            assert key in inference_outputs.keys()

        if n_samples > 1:
            expected_background_latent_shape = (
                n_samples,
                batch_size,
                n_background_latent,
            )
            expected_salient_latent_shape = (n_samples, batch_size, n_salient_latent)
            expected_library_shape = (n_samples, batch_size, 1)
        else:
            expected_background_latent_shape = (batch_size, n_background_latent)
            expected_salient_latent_shape = (batch_size, n_salient_latent)
            expected_library_shape = (batch_size, 1)

        assert inference_outputs["z"].shape == expected_background_latent_shape
        # assert inference_outputs["qz_m"].shape == expected_background_latent_shape
        # assert inference_outputs["qz_v"].shape == expected_background_latent_shape
        assert inference_outputs["s"].shape == expected_salient_latent_shape
        # assert inference_outputs["qs_m"].shape == expected_salient_latent_shape
        # assert inference_outputs["qs_v"].shape == expected_salient_latent_shape
        assert inference_outputs["library_rna"].shape == expected_library_shape

    def test_inference(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
    ):
        inference_input = mock_mocavi_module._get_inference_input(mock_mocavi_batch)
        inference_outputs = mock_mocavi_module.inference(**inference_input)
        for data_source in REQUIRED_DATA_SOURCES:
            assert data_source in inference_outputs.keys()
        background_s = inference_outputs["background"]["s"]

        # Background salient variables should be all zeros.
        assert torch.equal(background_s, torch.zeros_like(background_s))


class TestMoCAVIGenerative:
    def test_get_generative_input_from_concat_tensors(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
        mock_n_input,
    ):
        generative_input = mock_mocavi_module._get_generative_input_from_concat_tensors(mock_mocavi_batch, "background")
        for key in REQUIRED_GENERATIVE_INPUT_KEYS_FROM_CONCAT_TENSORS:
            assert key in generative_input.keys()
        batch_index = generative_input["batch_index"]
        assert batch_index.shape[1] == 1

    def test_get_generative_input_from_inference_outputs(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
    ):
        inference_outputs = mock_mocavi_module.inference(**mock_mocavi_module._get_inference_input(mock_mocavi_batch))
        generative_input = mock_mocavi_module._get_generative_input_from_inference_outputs(
            inference_outputs, REQUIRED_DATA_SOURCES[0]
        )
        for key in REQUIRED_GENERATIVE_INPUT_KEYS_FROM_INFERENCE_OUTPUTS:
            assert key in generative_input

        z = generative_input["z"]
        s = generative_input["s"]
        library_rna = generative_input["library_rna"]
        library_acc = generative_input["library_acc"]
        batch_size = z.shape[0]

        assert z.shape == (batch_size, mock_mocavi_module.n_background_latent)
        assert s.shape == (batch_size, mock_mocavi_module.n_salient_latent)
        assert library_rna.shape == (batch_size, 1)
        assert library_acc.shape == (batch_size, 1)

    def test_get_generative_input(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
        mock_background_batch,
        mock_target_batch,
    ):
        inference_outputs = mock_mocavi_module.inference(**mock_mocavi_module._get_inference_input(mock_mocavi_batch))
        generative_input = mock_mocavi_module._get_generative_input(mock_mocavi_batch, inference_outputs)
        for data_source in REQUIRED_DATA_SOURCES:
            assert data_source in generative_input.keys()
        background_generative_input = generative_input["background"]
        background_generative_input_keys = background_generative_input.keys()
        target_generative_input = generative_input["target"]
        target_generative_input_keys = target_generative_input.keys()
        for key in (
            REQUIRED_GENERATIVE_INPUT_KEYS_FROM_CONCAT_TENSORS + REQUIRED_GENERATIVE_INPUT_KEYS_FROM_INFERENCE_OUTPUTS
        ):
            assert key in background_generative_input_keys
            assert key in target_generative_input_keys

        # Check background vs. target labels are consistent.
        assert (background_generative_input["batch_index"] != mock_background_batch).sum() == 0
        assert (target_generative_input["batch_index"] != mock_target_batch).sum() == 0

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_generic_generative(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
        n_samples,
    ):
        inference_outputs = mock_mocavi_module.inference(
            **mock_mocavi_module._get_inference_input(mock_mocavi_batch),
            n_samples=n_samples,
        )
        generative_input = mock_mocavi_module._get_generative_input(mock_mocavi_batch, inference_outputs)["background"]
        generative_outputs = mock_mocavi_module._generic_generative(**generative_input)
        for key in REQUIRED_GENERATIVE_OUTPUT_KEYS:
            assert key in generative_outputs.keys()
        px_scale = generative_outputs["px_scale"]
        py_ = generative_outputs["py_"]
        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        batch_size = px_scale.shape[-2]
        n_input_genes = mock_mocavi_module.n_input_genes
        n_input_proteins = mock_mocavi_module.n_input_proteins

        if n_samples > 1:
            expected_shape_genes = (n_samples, batch_size, n_input_genes)
            expected_shape_proteins = (n_samples, batch_size, n_input_proteins)
        else:
            expected_shape_genes = (batch_size, n_input_genes)
            expected_shape_proteins = (batch_size, n_input_proteins)

        assert px_scale.shape == expected_shape_genes
        assert px_r.shape == (n_input_genes,)  # One dispersion parameter per gene.
        assert px_rate.shape == expected_shape_genes
        assert py_["scale"].shape == expected_shape_proteins

    def test_generative(
        self,
        mock_mocavi_module,
        mock_mocavi_batch,
    ):
        inference_outputs = mock_mocavi_module.inference(
            **mock_mocavi_module._get_inference_input(mock_mocavi_batch),
        )
        generative_input = mock_mocavi_module._get_generative_input(mock_mocavi_batch, inference_outputs)
        generative_outputs = mock_mocavi_module.generative(**generative_input)
        for data_source in REQUIRED_DATA_SOURCES:
            assert data_source in generative_outputs.keys()


class TestMoCAVILoss:
    def test_loss(self, mock_mocavi_module, mock_mocavi_vi_data):
        expected_shape = mock_mocavi_vi_data["inference_outputs"]["background"]["qz_m"].shape[:-1]
        losses = mock_mocavi_module.loss(
            mock_mocavi_vi_data["concat_tensors"],
            mock_mocavi_vi_data["inference_outputs"],
            mock_mocavi_vi_data["generative_outputs"],
        )
        loss = losses.loss
        recon_loss_sum = losses.reconstruction_loss_sum
        kl_local_sum = losses.kl_local_sum
        wasserstein_loss_sum = losses.extra_metrics["wasserstein_loss_sum"]

        assert loss.shape == torch.Size([])
        assert recon_loss_sum.shape == torch.Size([])
        assert kl_local_sum.shape == torch.Size([])
        assert wasserstein_loss_sum.shape == torch.Size([])

        recon_loss = losses.reconstruction_loss["reconstruction_loss"]
        kl_local = losses.kl_local

        assert recon_loss.shape == expected_shape
        for _, val in kl_local.items():
            assert val.shape == expected_shape

    @pytest.mark.parametrize("compute_loss", [True, False])
    def test_forward(
        self,
        mock_mocavi_module,
        mock_mocavi_vi_data,
        compute_loss,
    ):
        concat_tensors = mock_mocavi_vi_data["concat_tensors"]
        if compute_loss:
            inference_outputs, generative_outputs, losses = mock_mocavi_module(
                concat_tensors, compute_loss=compute_loss
            )
            assert isinstance(losses, LossOutput)
        else:
            inference_outputs, generative_outputs = mock_mocavi_module(concat_tensors, compute_loss=compute_loss)
        for data_source in REQUIRED_DATA_SOURCES:
            assert data_source in inference_outputs.keys()
            assert data_source in generative_outputs.keys()
            for key in REQUIRED_INFERENCE_OUTPUT_KEYS:
                assert key in inference_outputs[data_source].keys()
            for key in REQUIRED_GENERATIVE_OUTPUT_KEYS:
                assert key in generative_outputs[data_source].keys()
