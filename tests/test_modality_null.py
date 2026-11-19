import numpy as np
import pytest
import scipy.linalg
import torch
from scvi.distributions._negative_binomial import log_mixture_nb

from mocavi import MoCAVIModule
from mocavi import modality_null as mn

N_CELLS, D_Z, D_S = 4, 2, 3


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    module = MoCAVIModule(
        n_input_genes=20, n_input_regions=0, n_input_proteins=5, n_labels=2, n_batch=2,
        n_background_latent=D_Z, n_salient_latent=D_S, n_layers_decoder=2,
    ).eval().double()  # fmt: skip
    z, s = torch.randn(N_CELLS, D_Z, dtype=torch.float64), torch.randn(N_CELLS, D_S, dtype=torch.float64)
    batch = torch.tensor([[0], [1], [0], [1]])
    library = torch.full((N_CELLS,), 200.0, dtype=torch.float64)
    return module, z, s, batch, library


def random_psd(rng, d, rank=None):
    X = rng.standard_normal((d, rank or d))
    return X @ X.T


def rotate(rng, *Ms):
    Q = np.linalg.qr(rng.standard_normal((len(Ms[0]), len(Ms[0]))))[0]
    return Q, [Q @ M @ Q.T for M in Ms]


# ----------------------------------------------------------------------------- decoder and metrics


def test_decode_matches_module(setup):
    module, z, s, batch, library = setup
    log_rho, eta = mn.decode(module, z, s, batch)
    out = module._generic_generative(z, s, library.log()[:, None], library.log()[:, None], batch)
    py = out["py_"]
    torch.testing.assert_close(log_rho.exp(), out["px_scale"])
    torch.testing.assert_close(eta[..., 0], py["back_alpha"])
    torch.testing.assert_close((eta[..., 1] - eta[..., 0]).exp(), py["fore_scale"])
    torch.testing.assert_close(eta[..., 2], py["mixing"])


def test_jacobian_matches_autograd(setup):
    module, z, s, batch, _ = setup
    _, (J_R, J_P) = mn.jacobian(module, z, s, batch)
    for i in range(N_CELLS):
        ref_R, ref_P = torch.autograd.functional.jacobian(
            lambda s_: mn.decode(module, z[i : i + 1], s_, batch[i : i + 1]), s[i : i + 1]
        )
        torch.testing.assert_close(J_R[i], ref_R[0, :, 0])
        torch.testing.assert_close(J_P[i], ref_P[0, :, :, 0])


def kl_along(module, z, s, batch, library, v, y_max=400):
    """KL[p(. | s) || p(. | s + v)] per modality, written independently of the Fisher formulas."""
    (log_rho0, eta0), (log_rho1, eta1) = mn.decode(module, z, s, batch), mn.decode(module, z, s + v, batch)
    mu0, mu1, th = library * log_rho0.exp(), library * log_rho1.exp(), module.px_r.exp()
    kl_rna = (mu0 * torch.log(mu0 * (th + mu1) / (mu1 * (th + mu0))) + th * torch.log((th + mu1) / (th + mu0))).sum()

    y = torch.arange(y_max, dtype=torch.float64)[:, None]
    log_p = [
        log_mixture_nb(y, e[0, :, 0].exp(), e[0, :, 1].exp(), module.py_r.exp(), None, e[0, :, 2]) for e in (eta0, eta1)
    ]
    kl_pro = (log_p[0].exp() * (log_p[0] - log_p[1])).sum()
    return kl_rna, kl_pro


def test_metrics_are_kl_hessians(setup):
    """Eq. (1): G_m is the Hessian at v = 0 of v -> KL[p_m(. | s) || p_m(. | s + v)]."""
    module, z, s, batch, library = setup
    G_R, G_P, mass = mn.pullback_metrics(module, z, s, batch, library, q=1 - 1e-12)
    assert (mass > 1 - 1e-10).all()
    for i in range(N_CELLS):
        cell = (z[i : i + 1], s[i : i + 1], batch[i : i + 1], library[i : i + 1, None])
        for m, G in enumerate((G_R, G_P)):
            v0 = torch.zeros(1, D_S, dtype=torch.float64)
            H = torch.autograd.functional.hessian(lambda v: kl_along(module, *cell, v)[m], v0)  # noqa: B023
            torch.testing.assert_close(G[i], H[0, :, 0], rtol=1e-6, atol=1e-10)


def test_log_expression_matches_mocavi_fold_change_scale(setup):
    module, z, s, batch, library = setup
    py = module._generic_generative(z, s, library.log()[:, None], library.log()[:, None], batch)
    px_scale, py = py["px_scale"], py["py_"]
    foreground = torch.sigmoid(-py["mixing"]) * py["back_alpha"].exp() * py["fore_scale"]  # background at its mean
    log_expr = mn.log_expression(module, z, s, batch)
    torch.testing.assert_close(log_expr["rna"], torch.log2(1 + 1e4 * px_scale))
    torch.testing.assert_close(log_expr["protein"], torch.log2(0.5 + foreground))


def test_lfc_per_step_tends_to_derivative(setup):
    module, z, s, batch, _ = setup
    v, t = torch.randn_like(s), 1e-6  # small t: the ReLU decoders are only piecewise smooth
    L = mn.lfc(module, z, s, batch, v, t)
    for modality in mn.MODALITIES:
        f = lambda s_: mn.log_expression(module, z, s_, batch)[modality]  # noqa: E731, B023
        torch.testing.assert_close(L[modality] / t, torch.func.jvp(f, (s,), (v,))[1], rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------------- standardization


def test_support_standardizes_rank_deficient_covariance():
    rng = np.random.default_rng(0)
    s_target = np.c_[rng.standard_normal((500, 2)), np.zeros(500)] @ np.linalg.qr(rng.standard_normal((3, 3)))[0]
    C = mn.local_covariance(s_target, s_target[0], k=100)
    Q, sqrt_lam = mn.support(C)
    assert Q.shape == (3, 2)
    v = (Q * sqrt_lam) @ (a := rng.standard_normal(2)) / np.linalg.norm(a)
    np.testing.assert_allclose(v @ np.linalg.pinv(C) @ v, 1.0)


# ----------------------------------------------------------------------------- solvers


def test_rayleigh_quotient_full_rank():
    rng = np.random.default_rng(0)
    At, An = random_psd(rng, 5), random_psd(rng, 5)
    a, rec = mn.rayleigh_quotient(At, An)
    w, V = scipy.linalg.eigh(At, An)
    assert rec["status"] == "finite"
    np.testing.assert_allclose((a @ V[:, -1]) ** 2 / (V[:, -1] @ V[:, -1]), 1.0)
    np.testing.assert_allclose(a @ At @ a / (a @ An @ a), w[-1])
    np.testing.assert_allclose(rec["kappa"][0] / (1 - rec["kappa"][0]), w[-1])


def test_rayleigh_quotient_infinite_ratio_breaks_ties_by_target():
    rng = np.random.default_rng(0)
    Q, (At, An) = rotate(rng, np.diag([1.0, 3.0, 2.0, 0.5]), np.diag([0.0, 0.0, 1.0, 2.0]))
    a, rec = mn.rayleigh_quotient(At, An)
    assert rec["status"] == "infinite"
    np.testing.assert_allclose((a @ Q[:, 1]) ** 2, 1.0)  # largest target response within Null(An)


def test_subspace_projection_exact_branch():
    rng = np.random.default_rng(0)
    Q, (At, An) = rotate(rng, np.diag([1.0, 3.0, 2.0, 0.5]), np.diag([0.0, 0.0, 1.0, 2.0]))
    a, rec = mn.subspace_projection(At, An, alpha=0.01, beta=0.01)
    assert rec["status"] == "exact" and rec["k"] == 2
    np.testing.assert_allclose((a @ Q[:, 1]) ** 2, 1.0)


def test_subspace_projection_fallback_and_scale_invariance():
    rng = np.random.default_rng(0)
    Q, (At, An) = rotate(rng, np.diag([2.0, 1.0, 1.0, 1.0]), np.diag([1e-3, 1.0, 2.0, 3.0]))
    a, rec = mn.subspace_projection(At, An, alpha=0.01, beta=0.01)
    assert rec["status"] == "approximate" and rec["k"] == 1
    np.testing.assert_allclose((a @ Q[:, 0]) ** 2, 1.0)
    a_scaled, _ = mn.subspace_projection(1e-3 * At, 1e4 * An, alpha=0.01, beta=0.01)
    np.testing.assert_allclose((a @ a_scaled) ** 2, 1.0)


def test_subspace_projection_empty_when_null_space_is_target_inactive():
    rng = np.random.default_rng(0)
    _, (At, An) = rotate(rng, np.diag([1e-4, 1.0, 1.0, 1.0]), np.diag([0.0, 1.0, 2.0, 3.0]))
    a, rec = mn.subspace_projection(At, An, alpha=0.01, beta=0.01)
    assert a is None and rec["status"] == "empty"


def test_directions_are_oriented_and_standardized():
    rng = np.random.default_rng(0)
    G_t, G_n, C = random_psd(rng, 4), random_psd(rng, 4, rank=2), random_psd(rng, 4, rank=3)
    Q, sqrt_lam = mn.support(C)
    for rec in mn.directions(G_t, G_n, Q, sqrt_lam, mn.Config()).values():
        u, v = rec["u"], rec["v"]
        assert u[np.argmax(np.abs(u))] > 0
        np.testing.assert_allclose(v @ np.linalg.pinv(C) @ v, 1.0)
        np.testing.assert_allclose(v @ G_t @ v, rec["target_response"])  # v'G v = u'A u
