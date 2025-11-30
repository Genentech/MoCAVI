# MoCAVI

[![Tests][badge-tests]][link-tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/Genentech/MoCAVI/test.yaml?branch=main
[link-tests]: https://github.com/Genentech/MoCAVI/actions/workflows/test.yaml

## Multi-modal Contrastive Analysis of Perturbation Data

This package introduces MoCAVI (Multi-modal Contrastive Analysis via Variational Inference), a python package for analyzing multi-modal single-cell perturbation data, integrating information across genes, proteins, and chromatin accessibility. MoCAVI is implemented using scvi-tools.

We also include EB-MoCAVI, an extension of MoCAVI that enables embedding-based modeling of unseen perturbations using empirical Bayes. We demonstrate the application of EB-MoCAVI on a multi-modal single-cell perturbation dataset with paired gene expression and optical pooled screening (OPS) measurements.

### Background:

Current single-cell technologies enable the simultaneous measurement of multiple modalities, such as gene expression, protein levels, and chromatin accessibility. However, analyzing perturbation effects in these multi-modal datasets presents unique challenges. The effect size of perturbations on molecular profiles is often smaller than the inherent heterogeneity in the initial cell population, limiting the power of traditional analysis methods.
We present MoCAVI, a novel probabilistic method for multi-modal analysis of single-cell perturbation data that explicitly models both background cellular heterogeneity and perturbation-induced changes. MoCAVI builds upon the framework of contrastive analysis to isolate perturbation effects in a separate latent space from the background heterogeneity, demonstrating higher sensitivity in handling various modalities.

### Key Features of MoCAVI

1. **Multi-modal Integration:** MoCAVI can handle any subset of modalities including gene expression, protein measurements, and chromatin accessibility.

2. **Dual Latent Spaces:** As previous methods performing Contrastive Analysis, MoCAVI produces two distinct latent spaces - a background space capturing control population heterogeneity, and a salient space isolating perturbation effects.

3. **Environment-specific Modeling:** MoCAVI partitions the dataset into multiple environments, including a baseline (control) environment and separate environments for each perturbation.

4. **Flexible Probability Distributions:** The model employs appropriate probability distributions for each modality (e.g., negative binomial for gene expression, mixture of negative binomial for protein measurements).

5. **Amortized Variational Inference:** MoCAVI is fitted using amortized variational inference with modality-specific encoder networks.

By applying MoCAVI, researchers can gain deeper insights into perturbation effects across multiple molecular layers, while accounting for background cellular heterogeneity. The attached tutorial will guide you through the process of using MoCAVI to analyze your multi-modal single-cell perturbation data.

### Additional Features of EB-MoCAVI

EB-MoCAVI extends MoCAVI by enabling the modeling of unseen perturbations using empirical Bayes. By leveraging embeddings from an auxiliary assay (e.g., OPS), EB-MoCAVI can predict perturbation effects for perturbations not present in the training data.

### Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing miniforge, which is a minimal conda installer.

1. Create a new conda env

```bash
conda create -n mocavi python=3.10
conda activate mocavi
```

2. Clone the repo and move into the folder

```bash
git clone https://www.github.com/Genentech/MoCAVI.git
cd MoCAVI
```

3. Install package and dependencies for the tutorial

```bash
pip install .
pip install scanpy seaborn
```

### Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

### Citation

MoCAVI

> t.b.a

EB-MoCAVI

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/Genentech/MoCAVI/issues
