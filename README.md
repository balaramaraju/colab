# colab

A Colab notebook + helper that walks through where GPU memory goes in a
Transformer forward/backward pass — activations, weights, gradients, optimizer
state — and how to release it.

## Files

- `cache_mem_profiler.py` — the `CacheMemoryProfiler` helper (single source of truth).
- `UnderstandTransformerMemory.ipynb` — the Colab notebook (shared via Colab).

## How the notebook loads the helper

The notebook downloads `cache_mem_profiler.py` from this repo at runtime:

```python
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/balaramaraju/colab/main/cache_mem_profiler.py",
    "cache_mem_profiler.py",
)
```

Edit the helper here and viewers get the latest on their next run.

## Usage

```python
from cache_mem_profiler import CacheMemoryProfiler

_, cache = model.run_with_cache(tokens, remove_batch_dim=True)

prof = CacheMemoryProfiler(cache.cache_dict, n_layers=model.cfg.n_layers, model=model)

prof.print_table(layer=0)
prof.print_layer_table()
print(prof.summary())
prof.plot_histograms()
```

### Dependencies

- `torch`, `numpy`, `matplotlib`
- `transformer_lens` (used by the notebook)
