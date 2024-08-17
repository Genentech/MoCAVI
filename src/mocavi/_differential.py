import inspect
import logging
import warnings
from collections.abc import Callable, Sequence
from collections.abc import Iterable as IterableClass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scvi import REGISTRY_KEYS, settings
from scvi._types import Number
from scvi.data import AnnDataManager
from scvi.model.base._differential import (
    densify,
    describe_continuous_distrib,
    estimate_delta,
    estimate_pseudocounts_offset,
    pairs_sampler,
)
from scvi.model.base._utils import _fdr_de_prediction, _prepare_obs
from scvi.utils import track

logger = logging.getLogger(__name__)


def adt_raw_counts_properties(
    adata_manager: AnnDataManager,
    idx1: list[int] | np.ndarray,
    idx2: list[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Computes and returns some statistics on the raw counts of two sub-populations.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object setup with :class:`~scvi.model.TOTALVI`.
    idx1
        subset of indices describing the first population.
    idx2
        subset of indices describing the second population.

    Returns
    -------
    type
        Dict of ``np.ndarray`` containing, by pair (one for each sub-population),
        mean expression per gene, proportion of non-zero expression per gene, mean of normalized
        expression.
    """
    protein_exp = adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)

    nan = np.array([np.nan] * adata_manager.summary_stats.n_proteins)
    protein_exp = adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
    mean1_pro = np.asarray(protein_exp[idx1].mean(0))
    mean2_pro = np.asarray(protein_exp[idx2].mean(0))
    nonz1_pro = np.asarray((protein_exp[idx1] > 0).mean(0))
    nonz2_pro = np.asarray((protein_exp[idx2] > 0).mean(0))
    properties = {
        "raw_mean1": mean1_pro,
        "raw_mean2": mean2_pro,
        "non_zeros_proportion1": nonz1_pro,
        "non_zeros_proportion2": nonz2_pro,
        "raw_normalized_mean1": nan,
        "raw_normalized_mean2": nan,
    }
    return properties


def _de_core(
    adata_manager,
    model_fn,
    groupby,
    group1,
    group2,
    idx1,
    idx2,
    all_stats,
    all_stats_fn,
    col_names,
    mode,
    batchid1,
    batchid2,
    delta,
    batch_correction,
    fdr,
    silent,
    **kwargs,
):
    """Internal function for DE interface."""
    adata = adata_manager.adata
    if group1 is None and idx1 is None:
        group1 = adata.obs[groupby].astype("category").cat.categories.tolist()
        if len(group1) == 1:
            raise ValueError("Only a single group in the data. Can't run DE on a single group.")

    if not isinstance(group1, IterableClass) or isinstance(group1, str):
        group1 = [group1]

    # make a temp obs key using indices
    temp_key = None
    if idx1 is not None:
        obs_col, group1, group2 = _prepare_obs(idx1, idx2, adata)
        temp_key = "_scvi_temp_de"
        adata.obs[temp_key] = obs_col
        groupby = temp_key

    df_results = []
    dc = DifferentialComputation(
        model_fn=model_fn,  # representation_fn=None,
        adata_manager=adata_manager,
    )
    for g1 in track(
        group1,
        description="DE...",
        disable=silent,
    ):
        cell_idx1 = (adata.obs[groupby] == g1).to_numpy().ravel()
        if group2 is None:
            cell_idx2 = ~cell_idx1
        else:
            cell_idx2 = (adata.obs[groupby] == group2).to_numpy().ravel()

        all_info = dc.get_bayes_factors(
            cell_idx1,
            cell_idx2,
            mode=mode,
            delta=delta,
            batchid1=batchid1,
            batchid2=batchid2,
            use_observed_batches=not batch_correction,
            **kwargs,
        )

        if all_stats is True:
            genes_properties_dict = all_stats_fn(adata_manager, cell_idx1, cell_idx2)
            all_info = {**all_info, **genes_properties_dict}

        res = pd.DataFrame(all_info, index=col_names)
        sort_key = "proba_de" if mode == "change" else "bayes_factor"
        res = res.sort_values(by=sort_key, ascending=False)
        if mode == "change":
            res[f"is_de_fdr_{fdr}"] = _fdr_de_prediction(res["proba_de"], fdr=fdr)
        if idx1 is None:
            g2 = "Rest" if group2 is None else group2
            res["comparison"] = f"{g1} vs {g2}"
            res["group1"] = g1
            res["group2"] = g2
        df_results.append(res)

    if temp_key is not None:
        del adata.obs[temp_key]

    result = pd.concat(df_results, axis=0)

    return result


def _de_paired_test_core(
    adata_manager,
    model_fn1,
    model_fn2,
    groupby,
    group1,
    idx1,
    all_stats,
    all_stats_fn,
    col_names,
    mode,
    batchid1,
    delta,
    batch_correction,
    fdr,
    silent,
    **kwargs,
):
    """Internal function for DE interface. Adapted for paired testing with counterfactuals"""
    adata = adata_manager.adata
    if group1 is None and idx1 is None:
        group1 = adata.obs[groupby].astype("category").cat.categories.tolist()
        if len(group1) == 1:
            raise ValueError("Only a single group in the data. Can't run DE on a single group.")

    if not isinstance(group1, IterableClass) or isinstance(group1, str):
        group1 = [group1]

    # make a temp obs key using indices
    temp_key = None
    if idx1 is not None:
        obs_col, group1, group2 = _prepare_obs(idx1, idx1, adata)
        temp_key = "_scvi_temp_de"
        adata.obs[temp_key] = obs_col
        groupby = temp_key

    df_results = []
    dc = DifferentialComputation(
        model_fn=model_fn1,  # representation_fn=None,
        adata_manager=adata_manager,
    )
    dc.model_fn1 = model_fn1
    dc.model_fn2 = model_fn2
    for g1 in track(
        group1,
        description="DE...",
        disable=silent,
    ):
        cell_idx1 = (adata.obs[groupby] == g1).to_numpy().ravel()
        all_info = dc.get_synthetic_bayes_factors(
            cell_idx1,
            mode=mode,
            delta=delta,
            batchid1=batchid1,
            use_observed_batches=not batch_correction,
            **kwargs,
        )

        if all_stats is True:
            genes_properties_dict = all_stats_fn(adata_manager, cell_idx1, cell_idx1)
            all_info = {**all_info, **genes_properties_dict}

        res = pd.DataFrame(all_info, index=col_names)
        sort_key = "proba_de" if mode == "change" else "bayes_factor"
        res = res.sort_values(by=sort_key, ascending=False)
        if mode == "change":
            res[f"is_de_fdr_{fdr}"] = _fdr_de_prediction(res["proba_de"], fdr=fdr)
        if idx1 is None:
            g2 = "Vehicle"
            res["comparison"] = f"{g1} vs {g2}"
            res["group1"] = g1
            res["group2"] = g2
        df_results.append(res)

    if temp_key is not None:
        del adata.obs[temp_key]

    result = pd.concat(df_results, axis=0)

    return result


class DifferentialComputation:
    """
    Unified class for differential computation.

    This class takes a function from a model like `SCVI` or `TOTALVI` and takes outputs
    from this function with respect to the adata input and computed Bayes factors as
    described in :cite:p:`Lopez18`, :cite:p:`Xu21`, or :cite:p:`Boyeau19`.

    Parameters
    ----------
    model_fn
        Function in model API to get values from.
    adata_manager
        AnnDataManager created by :meth:`~scvi.model.SCVI.setup_anndata`.
    """

    def __init__(self, model_fn, adata_manager):
        self.adata_manager = adata_manager
        self.adata = adata_manager.adata
        self.model_fn = model_fn

    def get_bayes_factors(
        self,
        idx1: list[bool] | np.ndarray,
        idx2: list[bool] | np.ndarray,
        mode: Literal["vanilla", "change"] = "vanilla",
        batchid1: Sequence[Number | str] | None = None,
        batchid2: Sequence[Number | str] | None = None,
        use_observed_batches: bool | None = False,
        n_samples: int = 5000,
        use_permutation: bool = False,
        m_permutation: int = 10000,
        change_fn: str | Callable | None = None,
        m1_domain_fn: Callable | None = None,
        delta: float | None = 0.5,
        pseudocounts: float | None = 0.0,
        cred_interval_lvls: list[float] | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        r"""
        A unified method for differential expression inference.

        Two modes coexist:

        - The ``'vanilla'`` mode follows protocol described in :cite:p:`Lopez18` and :cite:p:`Xu21`.

            In this case, we perform hypothesis testing based on the hypotheses.

        .. math::
            M_1: h_1 > h_2 ~\text{and}~ M_2: h_1 \leq h_2.

        DE can then be based on the study of the Bayes factors

        .. math::
            \log p(M_1 | x_1, x_2) / p(M_2 | x_1, x_2).

        - The ``'change'`` mode (described in :cite:p:`Boyeau19`).

            This mode consists of estimating an effect size random variable (e.g., log fold-change) and
            performing Bayesian hypothesis testing on this variable. The `change_fn` function computes
            the effect size variable :math:`r` based on two inputs corresponding to the posterior quantities
            (e.g., normalized expression) in both populations.

        Hypotheses:

        .. math::
            M_1: r \in R_1 ~\text{(effect size r in region inducing differential expression)}

        .. math::
            M_2: r  \notin R_1 ~\text{(no differential expression)}

        To characterize the region :math:`R_1`, which induces DE, the user has two choices.

        1. A common case is when the region :math:`[-\delta, \delta]` does not induce differential
           expression. If the user specifies a threshold delta, we suppose that :math:`R_1 = \mathbb{R} \setminus [-\delta, \delta]`
        2. Specify an specific indicator function:

        .. math::
            f: \mathbb{R} \mapsto \{0, 1\} ~\text{s.t.}~ r \in R_1 ~\text{iff.}~ f(r) = 1.

        Decision-making can then be based on the estimates of

        .. math::
            p(M_1 \mid x_1, x_2).

        Both modes require to sample the posterior distributions.
        To that purpose, we sample the posterior in the following way:

        1. The posterior is sampled `n_samples` times for each subpopulation.
        2. For computational efficiency (posterior sampling is quite expensive), instead of
           comparing the obtained samples element-wise, we can permute posterior samples.
           Remember that computing the Bayes Factor requires sampling :math:`q(z_A \mid x_A)` and :math:`q(z_B \mid x_B)`.

        Currently, the code covers several batch handling configurations:

        1. If ``use_observed_batches=True``, then batch are considered as observations
           and cells' normalized means are conditioned on real batch observations.
        2. If case (cell group 1) and control (cell group 2) are conditioned on the same
           batch ids. This requires ``set(batchid1) == set(batchid2)`` or ``batchid1 == batchid2 === None``.
        3. If case and control are conditioned on different batch ids that do not intersect
           i.e., ``set(batchid1) != set(batchid2)`` and ``len(set(batchid1).intersection(set(batchid2))) == 0``.

        This function does not cover other cases yet and will warn users in such cases.

        Parameters
        ----------
        mode
            one of ["vanilla", "change"]
        idx1
            bool array masking subpopulation cells 1. Should be True where cell is
            from associated population
        idx2
            bool array masking subpopulation cells 2. Should be True where cell is
            from associated population
        batchid1
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 1. By default, all ids are taken into account
        batchid2
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 2. By default, all ids are taken into account
        use_observed_batches
            Whether posterior values are conditioned on observed
            batches
        n_samples
            Number of posterior samples
        use_permutation
            Activates step 2 described above.
            Simply formulated, pairs obtained from posterior sampling
            will be randomly permuted so that the number of pairs used
            to compute Bayes Factors becomes `m_permutation`.
        m_permutation
            Number of times we will "mix" posterior samples in step 2.
            Only makes sense when `use_permutation=True`
        change_fn
            function computing effect size based on both posterior values
        m1_domain_fn
            custom indicator function of effect size regions
            inducing differential expression
        delta
            specific case of region inducing differential expression.
            In this case, we suppose that :math:`R \setminus [-\delta, \delta]` does not induce differential expression
            (LFC case). If the provided value is `None`, then a proper threshold is determined
            from the distribution of LFCs accross genes.
        pseudocounts
            pseudocount offset used for the mode `change`.
            When None, observations from non-expressed genes are used to estimate its value.
        cred_interval_lvls
            List of credible interval levels to compute for the posterior
            LFC distribution

        Returns
        -------
        Differential expression properties

        """
        eps = 1e-8
        # Normalized means sampling for both populations
        scales_batches_1 = self.scale_sampler(
            selection=idx1,
            batchid=batchid1,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )
        scales_batches_2 = self.scale_sampler(
            selection=idx2,
            batchid=batchid2,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )

        px_scale_mean1 = scales_batches_1["scale"].mean(axis=0)
        px_scale_mean2 = scales_batches_2["scale"].mean(axis=0)

        # Sampling pairs
        # The objective of code section below is to ensure than the samples of normalized
        # means we consider are conditioned on the same batch id
        batchid1_vals = np.unique(scales_batches_1["batch"])
        batchid2_vals = np.unique(scales_batches_2["batch"])

        create_pairs_from_same_batches = (set(batchid1_vals) == set(batchid2_vals)) and not use_observed_batches
        if create_pairs_from_same_batches:
            # First case: same batch normalization in two groups
            logger.debug("Same batches in both cell groups")
            n_batches = len(set(batchid1_vals))
            n_samples_per_batch = m_permutation // n_batches if m_permutation is not None else None
            logger.debug(f"Using {n_samples_per_batch} samples per batch for pair matching")
            scales_1 = []
            scales_2 = []
            for batch_val in set(batchid1_vals):
                # Select scale samples that originate from the same batch id
                scales_1_batch = scales_batches_1["scale"][scales_batches_1["batch"] == batch_val]
                scales_2_batch = scales_batches_2["scale"][scales_batches_2["batch"] == batch_val]

                # Create more pairs
                scales_1_local, scales_2_local = pairs_sampler(
                    scales_1_batch,
                    scales_2_batch,
                    use_permutation=use_permutation,
                    m_permutation=n_samples_per_batch,
                )
                scales_1.append(scales_1_local)
                scales_2.append(scales_2_local)
            scales_1 = np.concatenate(scales_1, axis=0)
            scales_2 = np.concatenate(scales_2, axis=0)
        else:
            logger.debug("Ignoring batch conditionings to compare means")
            if len(set(batchid1_vals).intersection(set(batchid2_vals))) >= 1:
                warnings.warn(
                    "Batchids of cells groups 1 and 2 are different but have an non-null "
                    "intersection. Specific handling of such situations is not implemented "
                    "yet and batch correction is not trustworthy.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            scales_1, scales_2 = pairs_sampler(
                scales_batches_1["scale"],
                scales_batches_2["scale"],
                use_permutation=use_permutation,
                m_permutation=m_permutation,
            )

        # Adding pseudocounts to the scales
        if pseudocounts is None:
            logger.debug("Estimating pseudocounts offet from the data")
            x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
            where_zero_a = densify(np.max(x[idx1], 0)) == 0
            where_zero_b = densify(np.max(x[idx2], 0)) == 0
            pseudocounts = estimate_pseudocounts_offset(
                scales_a=scales_1,
                scales_b=scales_2,
                where_zero_a=where_zero_a,
                where_zero_b=where_zero_b,
            )
        logger.debug(f"Using pseudocounts ~ {pseudocounts}")
        # Core of function: hypotheses testing based on the posterior samples we obtained above
        if mode == "vanilla":
            logger.debug("Differential expression using vanilla mode")
            proba_m1 = np.mean(scales_1 > scales_2, 0)
            proba_m2 = 1.0 - proba_m1
            res = {
                "proba_m1": proba_m1,
                "proba_m2": proba_m2,
                "bayes_factor": np.log(proba_m1 + eps) - np.log(proba_m2 + eps),
                "scale1": px_scale_mean1,
                "scale2": px_scale_mean2,
            }

        elif mode == "change":
            logger.debug("Differential expression using change mode")

            # step 1: Construct the change function
            def lfc(x, y):
                return np.log2(x + pseudocounts) - np.log2(y + pseudocounts)

            if change_fn == "log-fold" or change_fn is None:
                change_fn = lfc
            elif not callable(change_fn):
                raise ValueError("'change_fn' attribute not understood")

            # step2: Construct the DE area function
            if m1_domain_fn is None:

                def m1_domain_fn(samples):
                    delta_ = delta if delta is not None else estimate_delta(lfc_means=samples.mean(0))
                    logger.debug(f"Using delta ~ {delta_:.2f}")
                    return np.abs(samples) >= delta_

            change_fn_specs = inspect.getfullargspec(change_fn)
            domain_fn_specs = inspect.getfullargspec(m1_domain_fn)
            if (len(change_fn_specs.args) != 2) | (len(domain_fn_specs.args) != 1):
                raise ValueError("change_fn should take exactly two parameters as inputs; m1_domain_fn one parameter.")
            try:
                change_distribution = change_fn(scales_1, scales_2)
                is_de = m1_domain_fn(change_distribution)
                delta_ = estimate_delta(lfc_means=change_distribution.mean(0)) if delta is None else delta
            except TypeError as err:
                raise TypeError(
                    "change_fn or m1_domain_fn have has wrong properties."
                    "Please ensure that these functions have the right signatures and"
                    "outputs and that they can process numpy arrays"
                ) from err
            proba_m1 = np.mean(is_de, 0)
            change_distribution_props = describe_continuous_distrib(
                samples=change_distribution,
                credible_intervals_levels=cred_interval_lvls,
            )
            change_distribution_props = {"lfc_" + key: val for (key, val) in change_distribution_props.items()}

            res = dict(
                proba_de=proba_m1,
                proba_not_de=1.0 - proba_m1,
                bayes_factor=np.log(proba_m1 + eps) - np.log(1.0 - proba_m1 + eps),
                scale1=px_scale_mean1,
                scale2=px_scale_mean2,
                pseudocounts=pseudocounts,
                delta=delta_,
                **change_distribution_props,
            )
        else:
            raise NotImplementedError(f"Mode {mode} not recognized")

        return res

    def get_synthetic_bayes_factors(
        self,
        idx1: list[bool] | np.ndarray,
        mode: Literal["vanilla", "change"] = "vanilla",
        batchid1: Sequence[Number | str] | None = None,
        use_observed_batches: bool | None = False,
        n_samples: int = 5000,
        change_fn: str | Callable | None = None,
        m1_domain_fn: Callable | None = None,
        delta: float | None = 0.5,
        pseudocounts: float | None = 0.0,
        cred_interval_lvls: list[float] | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        r"""
        A unified method for differential expression inference.

        Parameters
        ----------
        mode
            one of ["vanilla", "change"]
        idx1
            bool array masking subpopulation cells 1. Should be True where cell is
            from associated population
        batchid1
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 1. By default, all ids are taken into account
        use_observed_batches
            Whether posterior values are conditioned on observed
            batches
        n_samples
            Number of posterior samples
        change_fn
            function computing effect size based on both posterior values
        m1_domain_fn
            custom indicator function of effect size regions
            inducing differential expression
        delta
            specific case of region inducing differential expression.
            In this case, we suppose that :math:`R \setminus [-\delta, \delta]` does not induce differential expression
            (LFC case). If the provided value is `None`, then a proper threshold is determined
            from the distribution of LFCs accross genes.
        pseudocounts
            pseudocount offset used for the mode `change`.
            When None, observations from non-expressed genes are used to estimate its value.
        cred_interval_lvls
            List of credible interval levels to compute for the posterior
            LFC distribution

        Returns
        -------
        Differential expression properties

        """
        eps = 1e-8
        # raise an error if the field self.model_fn1 and self.model_fn2 are not defined
        if not hasattr(self, "model_fn1") or not hasattr(self, "model_fn2"):
            raise ValueError("model_fn1 and model_fn2 should be defined in the class to use this method")
        # Normalized means using both salient and background values
        self.model_fn = self.model_fn1
        scales_batches_1 = self.scale_sampler(
            selection=idx1,
            batchid=batchid1,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )
        px_scale_mean1 = scales_batches_1["scale"].mean(axis=0)

        # Nomalized means using only background values
        self.model_fn = self.model_fn2
        scales_batches_2 = self.scale_sampler(
            selection=idx1,
            batchid=batchid1,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )
        px_scale_mean2 = scales_batches_2["scale"].mean(axis=0)

        scales_1, scales_2 = scales_batches_1["scale"], scales_batches_2["scale"]
        # Adding pseudocounts to the scales
        if pseudocounts is None:
            logger.debug("Estimating pseudocounts offet from the data")
            x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
            where_zero_a = densify(np.max(x[idx1], 0)) == 0
            pseudocounts = estimate_pseudocounts_offset(
                scales_a=scales_1,
                scales_b=scales_2,
                where_zero_a=where_zero_a,
                where_zero_b=where_zero_a,
            )
        logger.debug(f"Using pseudocounts ~ {pseudocounts}")
        # Core of function: hypotheses testing based on the posterior samples we obtained above
        # CODE BELOW IS EXACTLY THE SAME AS get_bayes_factors
        if mode == "vanilla":
            logger.debug("Differential expression using vanilla mode")
            proba_m1 = np.mean(scales_1 > scales_2, 0)
            proba_m2 = 1.0 - proba_m1
            res = {
                "proba_m1": proba_m1,
                "proba_m2": proba_m2,
                "bayes_factor": np.log(proba_m1 + eps) - np.log(proba_m2 + eps),
                "scale1": px_scale_mean1,
                "scale2": px_scale_mean2,
            }

        elif mode == "change":
            logger.debug("Differential expression using change mode")

            # step 1: Construct the change function
            def lfc(x, y):
                return np.log2(x + pseudocounts) - np.log2(y + pseudocounts)

            if change_fn == "log-fold" or change_fn is None:
                change_fn = lfc
            elif not callable(change_fn):
                raise ValueError("'change_fn' attribute not understood")

            # step2: Construct the DE area function
            if m1_domain_fn is None:

                def m1_domain_fn(samples):
                    delta_ = delta if delta is not None else estimate_delta(lfc_means=samples.mean(0))
                    logger.debug(f"Using delta ~ {delta_:.2f}")
                    return np.abs(samples) >= delta_

            change_fn_specs = inspect.getfullargspec(change_fn)
            domain_fn_specs = inspect.getfullargspec(m1_domain_fn)
            if (len(change_fn_specs.args) != 2) | (len(domain_fn_specs.args) != 1):
                raise ValueError("change_fn should take exactly two parameters as inputs; m1_domain_fn one parameter.")
            try:
                change_distribution = change_fn(scales_1, scales_2)
                is_de = m1_domain_fn(change_distribution)
                delta_ = estimate_delta(lfc_means=change_distribution.mean(0)) if delta is None else delta
            except TypeError as err:
                raise TypeError(
                    "change_fn or m1_domain_fn have has wrong properties."
                    "Please ensure that these functions have the right signatures and"
                    "outputs and that they can process numpy arrays"
                ) from err
            proba_m1 = np.mean(is_de, 0)
            change_distribution_props = describe_continuous_distrib(
                samples=change_distribution,
                credible_intervals_levels=cred_interval_lvls,
            )
            change_distribution_props = {"lfc_" + key: val for (key, val) in change_distribution_props.items()}

            res = dict(
                proba_de=proba_m1,
                proba_not_de=1.0 - proba_m1,
                bayes_factor=np.log(proba_m1 + eps) - np.log(1.0 - proba_m1 + eps),
                scale1=px_scale_mean1,
                scale2=px_scale_mean2,
                pseudocounts=pseudocounts,
                delta=delta_,
                **change_distribution_props,
            )
        else:
            raise NotImplementedError(f"Mode {mode} not recognized")

        return res

    @torch.inference_mode()
    def scale_sampler(
        self,
        selection: list[bool] | np.ndarray,
        n_samples: int | None = 5000,
        n_samples_per_cell: int | None = None,
        batchid: Sequence[Number | str] | None = None,
        use_observed_batches: bool | None = False,
        give_mean: bool | None = False,
    ) -> dict:
        """
        Samples the posterior scale using the variational posterior distribution.

        Parameters
        ----------
        selection
            Mask or list of cell ids to select
        n_samples
            Number of samples in total per batch (fill either `n_samples_total`
            or `n_samples_per_cell`)
        n_samples_per_cell
            Number of time we sample from each observation per batch
            (fill either `n_samples_total` or `n_samples_per_cell`)
        batchid
            Biological batch for which to sample from.
            Default (None) sample from all batches
        use_observed_batches
            Whether normalized means are conditioned on observed
            batches or if observed batches are to be used
        give_mean
            Return mean of values


        Returns
        -------
        type
            Dictionary containing:
            `scale`
            Posterior aggregated scale samples of shape (n_samples, n_vars)
            where n_samples correspond to either:
            - n_bio_batches * n_cells * n_samples_per_cell
            or
            - n_samples_total
            `batch`
            associated batch ids

        """
        # Get overall number of desired samples and desired batches
        if batchid is None and not use_observed_batches:
            batch_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
            batchid = batch_registry.categorical_mapping
        if use_observed_batches:
            if batchid is not None:
                raise ValueError("Unconsistent batch policy")
            batchid = [None]
        if n_samples is None and n_samples_per_cell is None:
            n_samples = 5000
        elif n_samples_per_cell is not None and n_samples is None:
            n_samples = n_samples_per_cell * len(selection)
        if (n_samples_per_cell is not None) and (n_samples is not None):
            warnings.warn(
                "n_samples and n_samples_per_cell were provided. Ignoring n_samples_per_cell",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        n_samples = int(n_samples / len(batchid))
        if n_samples == 0:
            (
                warnings.warn(
                    "very small sample size, please consider increasing `n_samples`",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                ),
            )
            n_samples = 2

        # Selection of desired cells for sampling
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        selection = np.asarray(selection)
        if selection.dtype is np.dtype("bool"):
            if len(selection) < self.adata.shape[0]:
                raise ValueError("Mask must be same length as adata.")
            selection = np.asarray(np.where(selection)[0].ravel())

        # Sampling loop
        px_scales = []
        batch_ids = []
        for batch_idx in batchid:
            idx_selected = np.arange(self.adata.shape[0])[selection]
            px_scales.append(
                self.model_fn(
                    self.adata,
                    indices=idx_selected,
                    transform_batch=batch_idx,
                    n_samples_overall=n_samples,
                )
            )
            batch_idx = batch_idx if batch_idx is not None else np.nan
            batch_ids.append([batch_idx] * px_scales[-1].shape[0])
        px_scales = np.concatenate(px_scales)
        batch_ids = np.concatenate(batch_ids).reshape(-1)
        if px_scales.shape[0] != batch_ids.shape[0]:
            raise ValueError("sampled scales and batches have inconsistent shapes")
        if give_mean:
            px_scales = px_scales.mean(0)
        return {"scale": px_scales, "batch": batch_ids}
