# 70M · TPP20

Both methods use 74,325,248 parameters, 2836 optimizer steps, context 2048, global batch 256 and seed 42. The indexed corpus contains 5B training tokens with a separate validation prefix; these runs consume 1,486,848,768 training tokens.

- `train-loss.csv`: original per-step training losses and consumed tokens.
- `validation-loss.csv`: validation loss and perplexity at each evaluation.
- `70m-tpp20.json`: final validation metrics and experiment dimensions.

The figure uses a normalized exponential moving average with decay 0.997. Both curves use the same transformation; faint traces show the original losses. The plotted range focuses on loss 2.85–5.15. Final metrics in the table are unsmoothed validation values.

To regenerate:

```bash
pip install -r requirements-plot.txt
python scripts/plot_results.py
```
