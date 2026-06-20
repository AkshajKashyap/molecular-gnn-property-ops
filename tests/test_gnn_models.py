from torch_geometric.data import Batch

from molgnn_ops.featurization import featurize_smiles
from molgnn_ops.gnn_data import molecule_graph_to_pyg_data
from molgnn_ops.gnn_models import GCNRegressor, GINRegressor


def _tiny_batch() -> Batch:
    return Batch.from_data_list(
        [
            molecule_graph_to_pyg_data(featurize_smiles("CCO", target=1.0, split="train")),
            molecule_graph_to_pyg_data(featurize_smiles("Cl", target=2.0, split="train")),
        ]
    )


def test_gcn_regressor_forward_pass() -> None:
    batch = _tiny_batch()
    model = GCNRegressor(input_dim=22, hidden_dim=8, num_layers=2)
    output = model(batch)
    tensor_output = model(batch.x, batch.edge_index, batch.batch)

    assert output.shape == (2,)
    assert tensor_output.shape == (2,)


def test_gin_regressor_forward_pass() -> None:
    output = GINRegressor(input_dim=22, hidden_dim=8, num_layers=2)(_tiny_batch())

    assert output.shape == (2,)
