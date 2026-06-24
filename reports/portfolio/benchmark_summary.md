# Portfolio Benchmark Summary

This tracked snapshot records the verified ESOL scaffold-split results without copying large generated artifacts into Git.

| Method | Split | Seeds | Test RMSE mean | RMSE std | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Fingerprint random forest | scaffold | 42, 43, 44 | 1.8480 | 0.0214 | Morgan fingerprint classical baseline. |
| GCN | scaffold | 42, 43, 44 | 1.3395 | 0.0738 | Repeated-seed GNN comparison. |
| GIN | scaffold | 42, 43, 44 | 1.4499 | 0.1372 | Repeated-seed GNN comparison. |
| Promoted fixed-split GCN | scaffold | split 42, model 43 | 1.3502 | N/A | Selected by validation RMSE 1.3420. |

Test metrics are reported after validation-based model selection. They are not used to choose the promoted checkpoint.
