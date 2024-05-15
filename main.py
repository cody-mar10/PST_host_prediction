import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterator, Optional, Sequence

import lightning as L
import torch
from torch_geometric.data import HeteroData
from train import init_trainer, train

from data import DataConfig
from model import GNNConfig


@dataclass
class Trial:
    dropout: float
    encoder_n_layers: int
    decoder_hidden_dims: tuple[int, int]
    lr: float
    disjoint_train_ratio: float
    share_weights: bool


class Trialer:
    def __init__(
        self,
        dropout: Optional[Sequence[float]] = None,
        encoder_n_layers: Optional[Sequence[int]] = None,
        decoder_hidden_dims: Optional[Sequence[tuple[int, int]]] = None,
        lr: Optional[Sequence[float]] = None,
        disjoint_train_ratio: Optional[Sequence[float]] = None,
        test_share_weights: bool = True,
    ):
        if dropout is None:
            dropout = [0.25]

        if encoder_n_layers is None:
            encoder_n_layers = range(1, 4)

        if decoder_hidden_dims is None:
            decoder_hidden_dims_opts = [512, 256, 128, 64, 32]

            decoder_hidden_dims = [
                (dim1, dim2)
                for dim1, dim2 in product(decoder_hidden_dims_opts, repeat=2)
                if dim1 > dim2
            ]

        if lr is None:
            lr = [5e-4, 1e-3, 5e-3]

        if disjoint_train_ratio is None:
            disjoint_train_ratio = [0.0, 0.3]

        self.options = {
            "dropout": dropout,
            "encoder_n_layers": encoder_n_layers,
            "decoder_hidden_dims": decoder_hidden_dims,
            "lr": lr,
            "disjoint_train_ratio": disjoint_train_ratio,
        }

        if test_share_weights:
            share_weights = [True, False]
        else:
            share_weights = [True]

        self.options["share_weights"] = share_weights

    def __len__(self) -> int:
        n_opts = 1
        for values in self.options.values():
            n_opts *= len(values)

        return n_opts

    def __iter__(self) -> Iterator[Trial]:
        for values in product(*self.options.values()):
            yield Trial(*values)


CHERRY_TRIAL_VALUES = {
    # it's not actually clear how many conv layers cherry uses
    # since the paper suggests 2 layers, but the code shows 1
    "encoder_n_layers": [1, 2],
    "decoder_hidden_dims": [(128, 32)],
    "disjoint_train_ratio": [0.0],
    "test_share_weights": True,
}


OUTPUT_COLUMNS = [
    "dataset",
    # trialed values
    "dropout",
    "encoder_n_layers",
    "decoder_hidden_dims",
    "lr",
    "disjoint_train_ratio",
    "share_weights",
    # performance metrics
    "loss",
    "accuracy",
    "auprc",
    "n_parameters",
]


@dataclass
class Args:
    file: Path
    output: Path
    logdir: Path
    dataset: Optional[str]
    dummy: bool
    early_stopping: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Args":
        return cls(**vars(args))

    @classmethod
    def parse_args(cls) -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i",
            "--file",
            type=Path,
            metavar="FILE",
            required=True,
            help="Input heterogeneous PyG knowledge graph",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            metavar="FILE",
            required=True,
            help="Output tsv file for test dataset performance metrics",
        )
        parser.add_argument(
            "-l",
            "--logdir",
            type=Path,
            metavar="DIR",
            default=Path("lightning_root"),
            help="Lightning log dir for model checkpoints and training logs (default: %(default)s)",
        )
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            metavar="STR",
            help="Dataset name if cannot be inferred from file name",
        )
        parser.add_argument(
            "--dummy",
            action="store_true",
            help="Process entire graph at once instead of loading edges in batches",
        )
        parser.add_argument(
            "--early-stopping",
            action="store_true",
            help="Enable bad trial early stopping based on validation loss",
        )

        return cls.from_args(parser.parse_args())


def check_input_dim(file: Path) -> int:
    d: HeteroData = torch.load(file)
    return d.num_node_features["virus"]


def main():
    args = Args.parse_args()
    file = args.file

    in_dim = check_input_dim(args.file)

    if args.dataset is None:
        if "cherry" in file.name:
            dataset = "cherry"
        elif "pst-large" in file.name:
            dataset = "pst-large"
        elif "esm-large" in file.name:
            dataset = "esm-large"
        elif "kmer" in file.name:
            dataset = "kmer"
        else:
            raise ValueError(f"Unknown dataset: {file}")
    else:
        dataset = args.dataset

    if dataset == "cherry":
        trialer = Trialer(**CHERRY_TRIAL_VALUES)
    else:
        trialer = Trialer(test_share_weights=True)

    L.seed_everything(123)
    with open(args.output, "w") as fp:
        fp.write("\t".join(OUTPUT_COLUMNS) + "\n")
        for trial in trialer:
            if (in_dim, in_dim) < trial.decoder_hidden_dims:
                continue

            trainer = init_trainer(args.logdir)

            test_results = {
                "dataset": dataset,
                "n_parameters" "dropout": trial.dropout,
                "encoder_n_layers": trial.encoder_n_layers,
                "decoder_hidden_dims": trial.decoder_hidden_dims,
                "lr": trial.lr,
                "disjoint_train_ratio": trial.disjoint_train_ratio,
                "share_weights": trial.share_weights,
            }

            data_config = DataConfig.model_validate(
                {
                    "file": file,
                    "disjoint_train_ratio": trial.disjoint_train_ratio,
                    "dummy": args.dummy,
                }
            )

            model_config = GNNConfig.model_validate(
                {
                    "encoder": {
                        "in_dim": in_dim,
                        "dropout": trial.dropout,
                        "num_layers": trial.encoder_n_layers,
                    },
                    "decoder": {
                        "in_dim": in_dim,
                        "hidden_dims": trial.decoder_hidden_dims,
                        "dropout": trial.dropout,
                    },
                    "lr": trial.lr,
                    "share_weights": trial.share_weights,
                }
            )

            results = train(trainer, data_config, model_config)

            test_results.update(results)
            line = "\t".join(map(str, test_results.values()))
            fp.write(f"{line}\n")


if __name__ == "__main__":
    main()
