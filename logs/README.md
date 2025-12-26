# Training Logs

Real-time training logs for the cross-linguistic scaling experiment.

## Files

| File | Description |
|------|-------------|
| `en_125M_training.csv` | English 125M model training metrics (step, loss, perplexity) |
| `fr_125M_training.csv` | French 125M model training metrics (step, loss, perplexity) |
| `training_dual.csv` | Combined training log with both languages |
| `grammar_probes_en.csv` | English grammar probe results by checkpoint |
| `grammar_probes_fr.csv` | French grammar probe results by checkpoint |
| `validation_results.csv` | Validation perplexity on held-out data |
| `SCALING_ANALYSIS.md` | Analysis comparing results to Pythia and scaling laws |

## Key Results

- **French grammar saturation**: 100% accuracy at step 12k (~197M tokens), stable thereafter
- **English grammar**: Fluctuates around chance (40-50%) through step 181k (~3B tokens)
- **Perplexity ratio**: 50x at matched training steps (FR ~27 vs EN ~1340)
- **Estimated training efficiency**: 50-100x (cross-study comparison with Pythia)

## Pre-registration

This experiment was pre-registered on OSF: [10.17605/OSF.IO/SJ48B](https://osf.io/sj48b)
