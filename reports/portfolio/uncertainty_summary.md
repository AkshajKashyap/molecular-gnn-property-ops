# Portfolio Uncertainty Summary

The fixed-split ensemble experiment is retained because it is an important negative result: model disagreement did not rank ESOL prediction errors.

## Individual Fixed-Split Models

| Model seed | Validation RMSE | Test RMSE | Test MAE | Test R2 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 1.3704 | 1.4148 | 1.0862 | 0.6093 |
| 43 | 1.3420 | 1.3502 | 1.0385 | 0.6441 |
| 44 | 1.5397 | 1.7675 | 1.4332 | 0.3902 |

## Ensemble Metrics

- Ensemble test RMSE: 1.4497
- Ensemble test MAE: 1.1205

## Interval Calibration

| Nominal coverage | Empirical coverage | Mean interval width |
| ---: | ---: | ---: |
| 0.8000 | 0.8225 | 5.4297 |
| 0.9000 | 0.9349 | 9.7732 |
| 0.9500 | 0.9882 | 20.1541 |

## Error Ranking

- Pearson uncertainty-error correlation: -0.0158
- Rank uncertainty-error correlation: -0.0165

| Retained fraction | Test RMSE | Mean uncertainty |
| ---: | ---: | ---: |
| 0.2500 | 1.5232 | 0.1885 |
| 0.5000 | 1.5257 | 0.2631 |
| 0.7500 | 1.4574 | 0.3472 |
| 1.0000 | 1.4497 | 0.4883 |

## Conclusion

Ensemble disagreement was not a useful uncertainty signal for this ESOL setup. The API and dashboard therefore expose applicability context, not unsupported confidence estimates.
