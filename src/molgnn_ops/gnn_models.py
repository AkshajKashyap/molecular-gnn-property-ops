import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


def _validate_model_config(hidden_dim: int, num_layers: int, dropout: float) -> None:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be greater than 0")
    if num_layers <= 0:
        raise ValueError("num_layers must be greater than 0")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 and 1")


def _graph_inputs(data_or_x, edge_index=None, batch=None):
    if edge_index is None:
        x = data_or_x.x
        edge_index = data_or_x.edge_index
        batch = getattr(data_or_x, "batch", None)
    else:
        x = data_or_x
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    return x, edge_index, batch


class GCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_model_config(hidden_dim, num_layers, dropout)
        dimensions = [input_dim, *([hidden_dim] * num_layers)]
        self.convolutions = nn.ModuleList(
            GCNConv(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data_or_x, edge_index=None, batch=None) -> torch.Tensor:
        x, edge_index, batch = _graph_inputs(data_or_x, edge_index, batch)
        for convolution in self.convolutions:
            x = convolution(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, batch)
        return self.head(pooled).view(-1)


class GINRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_model_config(hidden_dim, num_layers, dropout)
        convolutions = []
        current_dim = input_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            convolutions.append(GINConv(mlp, train_eps=True))
            current_dim = hidden_dim
        self.convolutions = nn.ModuleList(convolutions)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data_or_x, edge_index=None, batch=None) -> torch.Tensor:
        x, edge_index, batch = _graph_inputs(data_or_x, edge_index, batch)
        for convolution in self.convolutions:
            x = convolution(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, batch)
        return self.head(pooled).view(-1)
