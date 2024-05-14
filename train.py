from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from data import DataConfig, GraphDataModule
from model import GNNConfig, GNNModule

FilePath = str | Path


def init_trainer(savedir: FilePath, **kwargs) -> L.Trainer:
    monitor = "val_loss"

    callbacks: list[Callback] = [
        EarlyStopping(
            monitor=monitor,
            min_delta=0.01,
            patience=5,
            stopping_threshold=0.1,
            divergence_threshold=1.0,
        ),
        ModelCheckpoint(
            monitor=monitor,
            filename="{epoch}_{val_loss:.3f}",
            save_last=True,
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=250,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        # precision="16-true",
        num_sanity_val_steps=0,
        default_root_dir=savedir,
        **kwargs
    )

    return trainer


def train(
    trainer: L.Trainer, data_config: DataConfig, model_config: GNNConfig
) -> dict[str, float]:
    datamodule = GraphDataModule(data_config)
    model = GNNModule(model_config, datamodule.data.metadata())

    trainer.fit(model, datamodule=datamodule)
    test_results = model.test_step(datamodule.data, negative_edge_sampling=True)

    test_results["n_parameters"] = model.num_parameters()

    return test_results
