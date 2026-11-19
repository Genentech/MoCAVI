"""Modality-specific directions analyses in the MoCAVI salient latent space.

python -m mocavi.modality_null
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import nbinom

from mocavi import MoCAVIModel

MODALITIES = ("rna", "protein")
EPS = np.finfo(np.float64).eps


@dataclass
class Config:
    adata_path: str = "data/human_icCITE_seq_filtered.h5ad"
    model_dir: str = "demos/data/model"
    out_dir: str = "results/modality_null"
    k_neighbors: int = 256
    library: float | None = None  # None: each cell's observed RNA library size; else a fixed reference
    quantile: float = 1 - 1e-6  # count truncation for the protein Fisher blocks
    alpha: float = 0.01  # subspace projection: nuisance-null share
    beta: float = 0.01  # subspace projection: target-active share
    t: float = 1.0  # LFC step along v, in local standard deviations
    n_features: int = 20
    batch_size: int = 64


@dataclass
class Data:
    module: torch.nn.Module  # float64, eval mode
    obs: pd.DataFrame
    genes: np.ndarray
    proteins: np.ndarray
    target: np.ndarray  # bool, non-baseline cells
    z: np.ndarray  # background posterior means
    s: np.ndarray  # salient posterior means
    batch: np.ndarray
    library: np.ndarray


# -------------------------------------- data --------------------------------------


def load(cfg: Config) -> Data:
    adata = anndata.read_h5ad(cfg.adata_path)
    adata.uns["adt_select_names"] = adata.obsm["adt_select"].columns.tolist()
    MoCAVIModel.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="treatment",
        layer="counts",
        protein_expression_obsm_key="adt_select",
        protein_names_uns_key="adt_select_names",
    )
    model = MoCAVIModel.load(cfg.model_dir, adata)
    z = model.get_latent_representation(representation_kind="background")
    s = model.get_latent_representation(representation_kind="salient")
    treatment = adata.obs["treatment"].astype(str)
    baseline = (treatment.str.split("_").str[0] == "Vehicle") | treatment.isin(adata.uns["all_to_remove"])
    return Data(
        module=model.module.eval().double(),
        obs=adata.obs,
        genes=adata.var_names.to_numpy(),
        proteins=np.asarray(adata.uns["adt_select_names"]),
        target=~baseline.to_numpy(),
        z=z.astype(np.float64),
        s=s.astype(np.float64),
        batch=adata.obs["_scvi_batch"].to_numpy(np.int64),
        library=np.asarray(adata.layers["counts"].sum(1), dtype=np.float64).ravel(),
    )


# -------------------------------------- decoder --------------------------------------


def decode(module, z, s, batch):
    """Deterministic likelihood parameters at latent (z, s) and batch.

    Returns log rho (cells, genes) and the protein parameters (cells, proteins, 3) = (log r_back, log r_fore, background mixing logit), with the background log-mean fixed at its mean alpha_p instead of sampled.
    """
    latent = torch.cat([z, s], -1)
    rna = module.decoder_expression
    log_rho = torch.log_softmax(rna.px_scale_decoder[0](rna.px_decoder(latent, batch)), -1)

    pro = module.decoder_pro
    log_back = pro.py_back_mean_log_alpha(torch.cat([pro.py_back_decoder(latent, batch), latent], -1), batch)
    fore_scale = pro.py_fore_scale_decoder(torch.cat([pro.py_fore_decoder(latent, batch), latent], -1), batch)
    logit = pro.py_background_decoder(torch.cat([pro.sigmoid_decoder(latent, batch), latent], -1), batch)
    return log_rho, torch.stack([log_back, log_back + torch.log(fore_scale + 1 + 1e-8), logit], -1)


def log_expression(module, z, s, batch):
    """Decoded expression of both modalities on MoCAVI's fold-change scale.

    RNA: log2(1 + 1e4 * rho); protein: log2(0.5 + foreground rate (1 - sigmoid(logit)) * r_fore).
    """
    log_rho, eta = decode(module, z, s, batch)
    return {
        "rna": torch.log2(1 + 1e4 * log_rho.exp()),
        "protein": torch.log2(0.5 + torch.sigmoid(-eta[..., 2]) * eta[..., 1].exp()),
    }


def jacobian(module, z, s, batch):
    """Decoder parameters and their Jacobians in s: (log_rho, eta), (J_R (B, G, d), J_P (B, P, 3, d)).

    One forward-mode pass over d copies of each cell, the k-th copy carrying the unit tangent e_k.
    """
    n, d = s.shape
    rep = lambda x: x.repeat_interleave(d, 0)  # noqa: E731
    tangents = torch.eye(d, dtype=s.dtype).repeat(n, 1)
    params, jets = torch.func.jvp(lambda s_: decode(module, rep(z), s_, rep(batch)), (rep(s),), (tangents,))
    return [p[::d] for p in params], [j.unflatten(0, (n, d)).movedim(1, -1) for j in jets]


# -------------------------------------- Fisher pullback --------------------------------------


def nb_mixture_fisher(eta, theta, q):
    """Fisher information of the NB mixture in eta = (log mu_back, log mu_fore, logit), shape (N, 3, 3).

    Exact expectation over counts 0..y_max, where y_max is the largest q-quantile of the foreground component; since mu_fore >= mu_back with a shared theta, it bounds the mixture's q-quantile. Also returns the probability mass covered by the grid.
    """
    mu, logit = eta[:, :2].exp(), eta[:, 2]  # (N, 2), (N,)
    y_max = nbinom.ppf(q, theta.item(), (theta / (theta + mu[:, 1])).detach().numpy()).max()
    y = torch.arange(int(y_max) + 1, dtype=eta.dtype)[:, None, None]  # (Y, 1, 1)

    log_nb = (
        theta * (theta.log() - (theta + mu).log())
        + y * (mu.log() - (theta + mu).log())
        + torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1)
    )  # (Y, N, 2)
    log_weights = torch.stack([F.logsigmoid(logit), F.logsigmoid(-logit)], -1)  # background, foreground
    log_joint = log_nb + log_weights
    log_p = torch.logsumexp(log_joint, -1)  # (Y, N)
    resp = (log_joint - log_p[..., None]).exp()  # component responsibilities

    score_rates = resp * theta * (y - mu) / (theta + mu)  # d log p / d (log mu_back, log mu_fore)
    score_logit = resp[..., 0] - torch.sigmoid(logit)  # d log p / d logit
    score = torch.cat([score_rates, score_logit[..., None]], -1)
    p = log_p.exp()
    return torch.einsum("yn,yni,ynj->nij", p, score, score), p.sum(0)


def pullback_metrics(module, z, s, batch, library, q):
    """Fisher pullback metrics G_R, G_P (B, d, d) in s, and the protein grid's minimum covered mass (B,)."""
    (log_rho, eta), (J_R, J_P) = jacobian(module, z, s, batch)

    mu, theta_r = library[:, None] * log_rho.exp(), module.px_r.exp()
    G_R = torch.einsum("bgi,bg,bgj->bij", J_R, mu * theta_r / (mu + theta_r), J_R)

    theta_p = module.py_r.exp()
    fisher, mass = zip(*(nb_mixture_fisher(eta[:, p], theta_p[p], q) for p in range(len(theta_p))))
    G_P = torch.einsum("bpki,bpkl,bplj->bij", J_P, torch.stack(fisher, 1), J_P)
    return G_R, G_P, torch.stack(mass, 1).min(1).values


# -------------------------------------- local standardization --------------------------------------


def local_covariance(s_target, s_cell, k):
    """Covariance (1/K, centered on the neighbor mean) of the K nearest target salient embeddings."""
    neighbors = s_target[np.argpartition(((s_target - s_cell) ** 2).sum(1), k - 1)[:k]]
    return np.cov(neighbors.T, bias=True)


def zero_tol(eigenvalues):
    """Numerical zero for a PSD spectrum: 10 * dim * eps * lambda_max."""
    return 10 * len(eigenvalues) * EPS * eigenvalues.max()


def support(C):
    """Orthonormal basis Q of Range(C) and the square-root eigenvalues on it."""
    lam, Q = np.linalg.eigh(C)
    keep = lam > zero_tol(lam)
    return Q[:, keep], np.sqrt(lam[keep])


# -------------------------------------- opt solvers --------------------------------------
""" Both act on supported response operators A_t, A_n (r x r, standardized coordinates) and return the unit direction a in those coordinates (None if empty) plus solver-specific diagnostics. """


def rayleigh_quotient(At, An):
    """Leading generalized eigenvector of the raw quotient a'At a / a'An a, via the pencil (At, At + An)."""
    lam, V = np.linalg.eigh(At + An)
    keep = lam > zero_tol(lam)  # drop directions neither modality responds to
    W = V[:, keep] / np.sqrt(lam[keep])  # whitens At + An on its support (equivalent to the Cholesky route)
    kappa, Y = np.linalg.eigh(W.T @ At @ W)
    kappa, X = kappa[::-1], (W @ Y)[:, ::-1]  # target shares, descending; generalized eigenvectors

    infinite = 1 - kappa <= 10 * len(kappa) * EPS
    if infinite[0]:  # target-active part of Null(An): break ties by target response
        N = np.linalg.qr(X[:, infinite])[0]
        a = N @ np.linalg.eigh(N.T @ At @ N)[1][:, -1]
    else:
        a = X[:, 0] / np.linalg.norm(X[:, 0])
    return a, {"status": "infinite" if infinite[0] else "finite", "kappa": kappa, "kappa_gap": kappa[0] - kappa[1]}


def subspace_projection(At, An, alpha, beta):
    """Top target eigenvector within a nuisance-null subspace: exact null first, else the alpha-share fallback."""
    lam_n, E = np.linalg.eigh(An)
    branches = {"exact": lam_n <= zero_tol(lam_n), "approximate": np.cumsum(lam_n) <= alpha * lam_n.sum()}
    for status, mask in branches.items():
        N = E[:, mask]
        lam_t, Y = np.linalg.eigh(N.T @ At @ N)
        if lam_t.size and lam_t[-1] > beta * np.trace(At):
            a = N @ Y[:, -1]
            break
    else:
        status, a = "empty", None
    return a, {"status": status, "nuisance_spectrum": lam_n, "target_eigenvalues": lam_t[::-1], "k": int(mask.sum())}


def directions(G_t, G_n, Q, sqrt_lam, cfg):
    """Leading target-specific direction by both solvers, on A_m = C^1/2 G_m C^1/2 restricted to Range(C).

    Returns {solver: record}; a found direction is oriented (largest standardized coordinate positive) and recorded as u (standardized) and v = C^1/2 u (latent), with its responses. Empty records carry no u, v.
    """
    R = Q * sqrt_lam  # C^1/2 Q
    At, An = R.T @ G_t @ R, R.T @ G_n @ R
    solutions = {
        "rayleigh_quotient": rayleigh_quotient(At, An),
        "subspace_projection": subspace_projection(At, An, cfg.alpha, cfg.beta),
    }
    records = {}
    for solver, (a, rec) in solutions.items():
        if a is not None:
            u = Q @ a
            sign = np.sign(u[np.argmax(np.abs(u))])
            a, u = sign * a, sign * u
            tt, tn = a @ At @ a, a @ An @ a
            rec |= {
                "u": u,
                "v": R @ a,
                "target_response": tt,
                "nuisance_response": tn,
                "target_share": tt / np.trace(At),
                "nuisance_share": tn / np.trace(An),
                "ratio": np.inf if rec["status"] == "infinite" else tt / tn,
            }
        records[solver] = rec
    return records


# -------------------------------------- features --------------------------------------


def lfc(module, z, s, batch, v, t):
    """Centered log2 fold change per step t along v, per modality: [f(s + t v) - f(s - t v)] / 2, f = log_expression."""
    up, down = log_expression(module, z, s + t * v, batch), log_expression(module, z, s - t * v, batch)
    return {m: (up[m] - down[m]) / 2 for m in MODALITIES}


def top_features(L, names, k):
    idx = np.argsort(-np.abs(L))[:k]
    return names[idx], L[idx]


def features(module, inputs, v, target, nuisance, names, cfg):
    """Top-K target features by |LFC| at step t, their overlap with the top-K at t/2, and both modalities' max |LFC|."""
    v = torch.as_tensor(v)[None]
    L = {m: x[0].numpy() for m, x in lfc(module, *inputs, v, cfg.t).items()}
    top, top_lfc = top_features(L[target], names, cfg.n_features)
    half = top_features(lfc(module, *inputs, v, cfg.t / 2)[target][0].numpy(), names, cfg.n_features)[0]
    return {
        "top_features": top,
        "top_lfc": top_lfc,
        "half_step_overlap": len(set(top) & set(half)) / cfg.n_features,
        "target_max_lfc": np.abs(L[target]).max(),
        "nuisance_max_lfc": np.abs(L[nuisance]).max(),
    }


# -------------------------------------- analysis --------------------------------------


def cell_tensors(data: Data, idx, library=None):
    """(z, s, batch, library) tensors for cells idx; library=None uses the observed library sizes."""
    lib = data.library[idx] if library is None else np.full(len(idx), float(library))
    return tuple(torch.as_tensor(x) for x in (data.z[idx], data.s[idx], data.batch[idx][:, None], lib))


def cell_records(module, inputs, G, C, names, cfg):
    """Records for one cell, for each (target modality, solver); inputs = (z, s, batch) of shape (1, ...)."""
    Q, sqrt_lam = support(C)
    records = []
    for target, nuisance in (MODALITIES, MODALITIES[::-1]):
        for solver, rec in directions(G[target], G[nuisance], Q, sqrt_lam, cfg).items():
            if "v" in rec:
                rec |= features(module, inputs, rec["v"], target, nuisance, names[target], cfg)
            records.append({"target": target, "solver": solver, "support_rank": len(sqrt_lam)} | rec)
    return records


def run(cfg: Config) -> pd.DataFrame:
    data = load(cfg)
    obs = data.obs
    cells = np.flatnonzero(data.target)
    s_target = data.s[cells]
    C = np.stack([local_covariance(s_target, s_cell, cfg.k_neighbors) for s_cell in s_target])
    names = {"rna": data.genes, "protein": data.proteins}

    records, G_rna, G_protein, protein_mass = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(cells), cfg.batch_size):
            idx = cells[start : start + cfg.batch_size]
            z, s, batch, lib = cell_tensors(data, idx, cfg.library)
            G_R, G_P, mass = (x.numpy() for x in pullback_metrics(data.module, z, s, batch, lib, cfg.quantile))
            for i, cell in enumerate(idx):
                inputs = (z[i : i + 1], s[i : i + 1], batch[i : i + 1])
                meta = {"cell": obs.index[cell], "treatment": obs["treatment"].iloc[cell], "protein_mass": mass[i]}
                G = {"rna": G_R[i], "protein": G_P[i]}
                records += [meta | r for r in cell_records(data.module, inputs, G, C[start + i], names, cfg)]
            G_rna.append(G_R)
            G_protein.append(G_P)
            protein_mass.append(mass)
            print(f"{start + len(idx)}/{len(cells)} cells")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_pickle(out / "records.pkl")
    np.savez(
        out / "operators.npz",
        cells=obs.index[cells].to_numpy(str),
        G_rna=np.concatenate(G_rna),
        G_protein=np.concatenate(G_protein),
        C=C,
        protein_mass=np.concatenate(protein_mass),
    )
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return df


if __name__ == "__main__":
    run(Config())
