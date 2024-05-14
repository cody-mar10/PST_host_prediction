from typing import Literal, Optional, Sequence, TypeVar

import lightning as L
import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP, GraphConv, MessagePassing, to_hetero
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision

ConvT = TypeVar("ConvT", bound=MessagePassing)
TensorDict = dict[str, Tensor]
MetadataT = tuple[list[str], list[EdgeType]]


class EncoderConfig(BaseModel):
    in_dim: int = Field(
        -1, description="input data dimension (default: -1 = autodetect)"
    )
    dropout: float = Field(0.2, description="parameter dropout rate during training")
    conv_layer_type: type[MessagePassing] = Field(
        GraphConv,
        description="name of torch_geometric.nn.MessagePassing layer that is suitable for bipartite graphs",
    )
    hidden_dims: Optional[Sequence[int]] = Field(
        None, description="optional list of hidden dimensions for each layer"
    )
    num_layers: Optional[int] = Field(
        None,
        description="optional number of convolution layers that do NOT reduce the size of the input",
    )

    @model_validator(mode="after")
    def check_dims(self):
        if self.hidden_dims is not None:
            self.num_layers = len(self.hidden_dims)
        elif self.num_layers is not None:
            self.hidden_dims = [self.in_dim] * self.num_layers
        else:
            raise ValueError("Must provide either hidden_dims or num_layers.")

        return self


class LinkDecoderConfig(BaseModel):
    in_dim: int = Field(
        -1, description="input data dimension (default: -1 = autodetect)"
    )
    hidden_dims: tuple[int, int] = Field(
        (128, 32), description="decoder 2-layer MLP hidden dimensions"
    )
    dropout: float = Field(0.2, description="parameter dropout rate during training")
    concat: bool = Field(
        False,
        description="whether to concat the virus and host node dimensions along an edge or just subtract them from each other",
    )


class GNNConfig(BaseModel):
    encoder: EncoderConfig
    decoder: LinkDecoderConfig
    share_weights: bool = Field(
        False, description="whether to share the weights per edge type"
    )
    lr: float = Field(1e-3, description="learning rate", ge=1e-5, le=0.1)
    weight_decay: float = Field(
        1e-2, description="optimizer L2 penalty", ge=1e-3, le=0.1
    )


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dropout: float,
        num_layers: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        conv_layer_type: type[ConvT] = GraphConv,
    ):
        super().__init__()

        if hidden_dims is not None:
            # override num_layers even if it is provided
            num_layers = len(hidden_dims)
        elif num_layers is not None:
            hidden_dims = [in_dim] * num_layers
        else:
            raise ValueError("Either num_layers or hidden_dims must be provided")

        if num_layers < 1:
            raise ValueError(
                f"num_layers must be a positive integer, received {num_layers}"
            )

        if num_layers > 5:
            raise ValueError(
                f"num_layers must be less than or equal to 5, received {num_layers}"
            )

        self.actv = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for out_dim in hidden_dims:
            self.layers.append(
                conv_layer_type(
                    in_channels=(in_dim, in_dim),  # type: ignore
                    out_channels=out_dim,  # type: ignore
                    aggr="mean",
                )
            )

            in_dim = out_dim

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = self.actv(x)

        x = self.dropout(x)

        return x

    @classmethod
    def hetero(
        cls,
        metadata: MetadataT,
        aggr: Literal["sum", "mean", "min", "max", "mul"] = "sum",
        **init_kwargs,
    ) -> "Encoder":
        instance = cls(**init_kwargs)
        return to_hetero(module=instance, metadata=metadata, aggr=aggr)


class LinkDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, int],
        dropout: float,
        concat: bool = False,
    ):
        super().__init__()

        self.concat = concat
        hidden_dim1, hidden_dim2 = hidden_dims
        self.lin = MLP(
            channel_list=[in_dim * 2 if concat else in_dim, hidden_dim1, hidden_dim2],
            norm=None,
            dropout=dropout,
            plain_last=False,
            act="relu",
        )
        self.lin_pred = nn.Linear(hidden_dim2, 1)

    def forward(self, x_dict: TensorDict, edge_label_index: Tensor) -> Tensor:
        virus_idx, host_idx = edge_label_index

        virus_x = x_dict["virus"][virus_idx]
        host_x = x_dict["host"][host_idx]

        x: Tensor
        if self.concat:
            x = torch.cat([virus_x, host_x], dim=-1)
        else:
            # this is what CHERRY does
            x = virus_x - host_x
        x = self.lin(x)
        x = self.lin_pred(x)
        return x.squeeze()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_pred.reset_parameters()


class GNN(nn.Module):
    def __init__(self, config: GNNConfig, metadata: MetadataT):
        super().__init__()
        self.config = config
        self.encoder = Encoder.hetero(metadata=metadata, **config.encoder.model_dump())

        if config.share_weights:
            self.share_weights(("virus", "infects", "host"))

        self.decoder = LinkDecoder(**config.decoder.model_dump())

    def share_weights(self, reference_edge: EdgeType):
        """Share weights among each edge type."""
        src_edge = "__".join(reference_edge)
        src_params = {
            name: param
            for name, param in self.encoder.named_parameters()
            if src_edge in name and param.requires_grad
        }

        for name, param in self.encoder.named_parameters():
            if param.requires_grad and name not in src_params:
                fields = name.split(".")
                layer_no = int(fields[1])
                tgt_edge = fields[2]
                specific_name = fields[3]
                pname = fields[4]
                fields[2] = src_edge
                wanted = ".".join(fields)
                src_param = src_params[wanted]

                conv_layer: ConvT = self.encoder.layers[layer_no][tgt_edge]  # type: ignore
                attr = getattr(conv_layer, specific_name)
                setattr(attr, pname, src_param)

    def forward(
        self,
        x_dict: TensorDict,
        edge_index_dict: dict[EdgeType, Tensor],
        edge_label_index: torch.Tensor,
    ) -> Tensor:
        x_dict = self.encoder(x_dict, edge_index_dict)
        output = self.decoder(x_dict, edge_label_index)
        return output


class GNNModule(L.LightningModule):
    def __init__(self, config: GNNConfig, metadata: MetadataT):
        super().__init__()
        self.config = config
        self.model = GNN(config, metadata)
        self.criterion = nn.BCEWithLogitsLoss()

        # metrics
        self.auprc = BinaryAveragePrecision()
        self.accuracy = BinaryAccuracy()

        self.save_hyperparameters(config.model_dump())

    def configure_optimizers(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def forward(self, data: HeteroData, edge_label_index: Tensor) -> TensorDict:
        # The edge_index is used for message passing
        # While the edge_label_index is used for supervision / gets decoded
        return self.model(data.x_dict, data.edge_index_dict, edge_label_index)

    def train_val_step(self, batch: HeteroData, batch_idx: int, stage: str) -> Tensor:
        pred = self(batch, batch["infects"].edge_label_index)
        targets = batch["infects"].edge_label

        mask = batch["virus"].train_mask[batch["infects"].edge_label_index[0]]

        pred = pred[mask]
        targets = targets[mask]

        ### log metrics
        batch_size = batch["infects"].edge_label.numel()
        loss = self.criterion(pred, targets)
        accuracy = self.accuracy(pred, targets.long())

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_acc": accuracy,
            },
            prog_bar=True,
            batch_size=batch_size,
            logger=True,
            on_step=None,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        return self.train_val_step(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self.log("train_accuracy_epoch", self.accuracy, prog_bar=True)

    def validation_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        return self.train_val_step(batch, batch_idx, "val")

    @torch.no_grad()
    def test_step(
        self,
        batch: HeteroData,
        batch_idx: Optional[int] = None,
        negative_edge_sampling: bool = True,
    ) -> dict[str, float]:
        if negative_edge_sampling:
            # this will generate the same number of negative edges as positive edges
            # this is necessary since only predicting with the positive edges will
            # lead to artificially inflated metrics
            negative_edge_index = negative_sampling(
                edge_index=batch["infects"].edge_index,
                num_nodes=(batch["virus"].x.size(0), batch["host"].x.size(0)),
            )

            edge_label_index = torch.cat(
                (batch["infects"].edge_index, negative_edge_index), dim=-1
            )
            edge_label = torch.tensor([1, 0]).repeat_interleave(
                edge_label_index.size(1) // 2
            )

            assert edge_label.size(0) == edge_label_index.size(1)
        else:
            edge_label_index = batch["infects"].edge_index
            edge_label = torch.ones((edge_label_index.size(1),), dtype=torch.long)

        pred = self(batch, edge_label_index).squeeze()
        mask = batch["virus"].test_mask[edge_label_index[0]]

        pred = pred[mask]
        targets = edge_label[mask]

        results: dict[str, float] = {
            "loss": self.criterion(pred, targets.float()).item(),
            "accuracy": self.accuracy(pred, targets).item(),
            "auprc": self.auprc(pred, targets).item(),
        }

        return results

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
