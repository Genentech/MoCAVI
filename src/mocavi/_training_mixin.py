"""
Mixin classes for pre-coded features.

For more details on Mixin classes, see
https://docs.scvi-tools.org/en/0.9.0/user_guide/notebooks/model_user_guide.html#Mixing-in-pre-coded-features
"""

import numpy as np
import torch
from scvi.train import TrainingPlan, TrainRunner

from ._data_splitting import MoCAVIDataSplitter


class MoCAVITrainingMixin:
    """General methods for contrastive learning."""

    def train(
        self,
        background_indices: list[int],
        target_indices: list[int],
        max_epochs: int | None = None,
        accerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ) -> None:
        """
        Train a contrastive model.

        Args:
        ----
            background_indices: Indices for background samples in `adata`.
            target_indices: Indices for target samples in `adata`.
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            batch_size: Mini-batch size to use during training.
            early_stopping: Perform early stopping. Additional arguments can be passed
                in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
            plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
                arguments passed to `train()` will overwrite values present
                in `plan_kwargs`, when appropriate.
            **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
            None. The model is trained.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        data_splitter = MoCAVIDataSplitter(
            self.adata_manager,
            background_indices,
            target_indices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        # set the reweighting mode
        if self.reweighting_mode == "auto":
            pass
        elif self.reweighting_mode == "bt":
            # background target balancing mode!
            p = len(background_indices) / (len(background_indices) + len(target_indices))
            self.module.background_weight = torch.tensor(p)
        else:
            raise ValueError("Cannot train, ill-defined reweighint mode")

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()
