from typing import Literal, Optional

import lightning as L
import torch
from pydantic import BaseModel, Field, FilePath
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.typing import EdgeType


class DataConfig(BaseModel):
    file: FilePath = Field(description="Heterogeneous graph file saved by torch.save")
    batch_size: int = Field(
        128, description="batch size (number of edges)", ge=64, le=4096
    )
    num_val: float = Field(
        0.2,
        description="ratio of training edges to be used for validation during training",
        gt=0.0,
        le=0.25,
    )
    disjoint_train_ratio: float = Field(
        0.0,
        description="ratio of training edges to be used exclusively for evaluation and not message passing",
        ge=0.0,
        le=0.3,
    )
    bipartite_edge_sampling: list[int] = Field(
        [10, 10],
        description="number of edges to sample over n iterations (length of list) for bipartite edges",
    )
    same_src_edge_sampling: list[int] = Field(
        [30, 30],
        description="number of edges to sample over n iterations (length of list) for single source edges that connect the same node types",
    )


class GraphDataModule(L.LightningDataModule):
    data: HeteroData
    train_data: HeteroData
    val_data: HeteroData

    TEST_EDGE: EdgeType = ("virus", "infects", "host")
    REV_TEST_EDGE: EdgeType = ("host", "rev_infects", "virus")
    NODES = {"virus", "host"}
    EDGES = {
        ("virus", "infects", "host"),
        ("host", "rev_infects", "virus"),
        ("virus", "related_to", "virus"),
    }

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

        self.save_hyperparameters(config.model_dump())

        self.data = torch.load(config.file)

        if not isinstance(self.data, HeteroData):
            raise ValueError("Data must be a HeteroData object.")

        if not self.data.is_undirected:
            self.data = ToUndirected()(self.data)

        # how make this configurable...?
        num_neighbors: dict[EdgeType, list[int]] = {}
        for edge in self.data.metadata()[1]:
            src, _, dst = edge

            if src == dst:
                sampling_info = config.same_src_edge_sampling
            else:
                sampling_info = config.bipartite_edge_sampling

            num_neighbors[edge] = sampling_info

        self.dataloader_kwargs = dict(
            num_neighbors=num_neighbors,
            batch_size=config.batch_size,
            subgraph_type="induced",
        )

        self.neg_sampling_ratio = 1.0

    def setup(self, stage: Optional[Literal["fit", "test"]] = None):
        if stage == "fit" or stage is None:
            splitter = RandomLinkSplit(
                num_val=self.config.num_val,
                num_test=0.0,  # we already have designated test data
                is_undirected=True,
                neg_sampling_ratio=self.neg_sampling_ratio,
                disjoint_train_ratio=self.config.disjoint_train_ratio,
                edge_types=[self.TEST_EDGE],
                rev_edge_types=[self.REV_TEST_EDGE],
                add_negative_train_samples=False,
            )
            self.train_data, self.val_data, _ = splitter(self.data)
        elif stage == "test":
            # should probably negatively sample things here?
            self.test_data = self.data

    def train_dataloader(self) -> LinkNeighborLoader:
        train_dataloader = LinkNeighborLoader(
            self.train_data,
            **self.dataloader_kwargs,  # type: ignore
            edge_label_index=(
                self.TEST_EDGE,
                self.train_data[self.TEST_EDGE].edge_label_index,
            ),
            edge_label=self.train_data[self.TEST_EDGE].edge_label,
            neg_sampling_ratio=self.neg_sampling_ratio,
            shuffle=True,
        )

        return train_dataloader

    def val_dataloader(self) -> LinkNeighborLoader:
        val_dataloader = LinkNeighborLoader(
            self.val_data,
            **self.dataloader_kwargs,  # type: ignore
            edge_label_index=(
                self.TEST_EDGE,
                self.val_data[self.TEST_EDGE].edge_label_index,
            ),
            edge_label=self.val_data[self.TEST_EDGE].edge_label,
            neg_sampling_ratio=None,
            shuffle=False,
        )

        return val_dataloader

    @property
    def node_dim(self) -> int:
        return int(self.data["virus"].x.size(-1))

    @property
    def num_nodes(self) -> dict[str, int]:
        self.data.num_nodes
        return {node: int(self.data[node].x.size(0)) for node in self.NODES}
