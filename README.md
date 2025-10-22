# mocavi

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/Genentech/MoCAVI/test.yaml?branch=main
[link-tests]: https://github.com/Genentech/MoCAVI/actions/workflows/test.yml

## Multi-modal Contrastive Analysis of Perturbation Data

This package introduces MoCAVI (Multi-modal Contrastive Analysis via Variational Inference), a python package for analyzing multi-modal single-cell perturbation data, integrating information across genes, proteins, and chromatin accessibility. MoCAVI is implemented using scvi-tools.

# Background:

Current single-cell technologies enable the simultaneous measurement of multiple modalities, such as gene expression, protein levels, and chromatin accessibility. However, analyzing perturbation effects in these multi-modal datasets presents unique challenges. The effect size of perturbations on molecular profiles is often smaller than the inherent heterogeneity in the initial cell population, limiting the power of traditional analysis methods.
We present MoCAVI, a novel probabilistic method for multi-modal analysis of single-cell perturbation data that explicitly models both background cellular heterogeneity and perturbation-induced changes. MoCAVI builds upon the framework of contrastive analysis to isolate perturbation effects in a separate latent space from the background heterogeneity, demonstrating higher sensitivity in handling various modalities.

## Key Features of MoCAVI

1. **Multi-modal Integration:** MoCAVI can handle any subset of modalities including gene expression, protein measurements, and chromatin accessibility.

2. **Dual Latent Spaces:** As previous methods performing Contrastive Analysis, MoCAVI produces two distinct latent spaces - a background space capturing control population heterogeneity, and a salient space isolating perturbation effects.

3. **Environment-specific Modeling:** MoCAVI partitions the dataset into multiple environments, including a baseline (control) environment and separate environments for each perturbation.

4. **Flexible Probability Distributions:** The model employs appropriate probability distributions for each modality (e.g., negative binomial for gene expression, mixture of negative binomial for protein measurements).

5. **Amortized Variational Inference:** MoCAVI is fitted using amortized variational inference with modality-specific encoder networks.

By applying MoCAVI, researchers can gain deeper insights into perturbation effects across multiple molecular layers, while accounting for background cellular heterogeneity. The attached tutorial will guide you through the process of using MoCAVI to analyze your multi-modal single-cell perturbation data.

## Installation

You first need to clone the repository from GitHub:

```
git clone [https://github.com/Genentech/MoCAVI.git](https://github.com/Genentech/MoCAVI.git)
```


You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing miniforge, which is a minimal conda installer.


### Installation using `pixi`

#### 1\. Install `pixi`

If you don't already have `pixi` installed, use the following command to install it:

```bash
curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | sh
```

Restart your terminal after the installation for the `pixi` command to be available in your system's PATH.

#### 2\. Set Up the Project Environment

First, define an environment variable pointing to the location of your local `MoCAVI` repository. *Make sure to change the path in the command below* to match the location on your system.

```bash
# !! IMPORTANT: Change this path to the root directory of your local MoCAVI clone
export mocavi_path="~/tools/MoCAVI"
```

Next, navigate to the directory where you want to store your environment files and initialize the `pixi` project:

```bash
# We recommend creating a dedicated directory (e.g., ~/scratch/pixi/)
mkdir -p ~/scratch/pixi/
cd ~/scratch/pixi/

# Initialize the new environment 
pixi init mocavi
            
# Enter the newly created project directory
cd mocavi
```

You will run all subsequent commands from within this `mocavi` directory.

#### 3\. Add Dependencies

We will now add the required packages to the project. `pixi` will automatically manage and resolve the dependencies.

1.  **Add Python:**

    ```bash
    pixi add python=3.10
    ```

2.  **Add MoCAVI:**
    This command installs `MoCAVI` from your local path in "editable" mode (`--editable`). This means any changes you make to the `MoCAVI` source code will be immediately reflected in your environment.

    ```bash
    pixi add --pypi "MoCAVI @ file://$(realpath ${mocavi_path})" --editable
    ```

3.  **Add Tutorial & Notebook Dependencies:**
    Install `seaborn`, `scanpy`, and also `ipykernel` if you want to use this environment in Jupyter.

    ```bash
    pixi add seaborn scanpy ipykernel
    ```

#### 4\. Install and Activate the Environment

Run `pixi install` to ensure all packages are correctly installed based on the lock file.

```bash
pixi install
```

To activate and enter the environment, run:

```bash
pixi shell
```

Your terminal prompt should now change, indicating you are inside the active `mocavi` environment.


### Installation with `conda` and `pip`

1. Create a new conda environment and activate the  environment

```bash
conda create -n mocavi python=3.10
conda activate mocavi
```

2. Move into the folder where the repository was cloned

```bash
cd MoCAVI
```

3. Install in editable mode

```bash
pip install .
```

5. Install the dependencies for the tutorial

```bash
pip install seaborn scanpy 
```

### (Optional) Register the MoCAVI environment for Jupyter Notebook

If you plan to use this environment in Jupyter Lab or Jupyter Notebook, run the following command while the environment is active (i.e., after running `pixi shell` or `conda activate mocavi`):

```bash
python -m ipykernel install --user --name mocavi --display-name "MoCAVI Environment (mocavi)"
```

This makes the `"MoCAVI Environment (mocavi)"` kernel available in your Jupyter interface, allowing you to easily select it when running notebooks.


### Download the data used in the tutorial

```bash
mkdir data && cd data
wget "https://figshare.com/ndownloader/files/55582163?private_link=49ffd57d7316562c54aa" -O human_icCITE_seq_filtered.h5ad
cd ..
```

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/Genentech/MoCAVI/issues
