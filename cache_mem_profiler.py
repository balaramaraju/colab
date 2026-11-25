"""
cache_mem_profiler.py
---------------------
Memory profiler for Transformer caches produced by `model.run_with_cache`.

Usage
-----
    from cache_mem_profiler import CacheMemoryProfiler

    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    prof = CacheMemoryProfiler(cache, n_layers=model.cfg.n_layers, model=model)

    prof.print_table()                     # all tensors, sorted by size
    prof.print_table(layer=0)              # only layer 0
    prof.print_table(kind="activation", layer=0)
    prof.print_layer_table()               # layer x kind matrix
    print(prof.summary())                  # dict of MB per kind
    prof.plot_histograms()                 # all-layer histograms
    prof.plot_histograms(layer=0)          # layer-0 histograms only
    prof.plot_histograms(layer=[0, 2, 5])  # selected layers

Notes
-----
* Activations come from the cache (run_with_cache).
* Weights come from model.named_parameters().
* Gradients come from p.grad (call after loss.backward()).
* Optimizer state comes from the model's optimizer (call after optimizer.step()).
* Optimizer is looked up on `model.optimizer`, `model._optimizer`,
  `model.cfg.optimizer`, `model.cfg._optimizer`, or `model.trainer.optimizer`.
* The cache walker is defensive: it accepts a flat dict, nested dict/list,
  a wrapper exposing `.cache` or `.caching`, or a bare Tensor, and recursively
  flattens to (name, tensor) pairs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch

MB = 1024 ** 2


# ---------------------------------------------------------------------------
# Recursive tensor walker
# ---------------------------------------------------------------------------

def _iter_tensors(obj, prefix: str = ""):
    """Yield (name, tensor) recursively from dicts / lists / tuples / sets / Tensor."""
    if isinstance(obj, torch.Tensor):
        yield prefix.rstrip("."), obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}{k}." if prefix else f"{k}."
            yield from _iter_tensors(v, name)
        return
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{prefix}{i}.")
        return
    # anything else (None, numbers, configs) — silently skip
    return


# ---------------------------------------------------------------------------
# Tensor record
# ---------------------------------------------------------------------------

@dataclass
class TensorEntry:
    name: str
    shape: tuple
    dtype: torch.dtype
    nbytes: int
    kind: str  # 'weight' | 'activation' | 'gradient' | 'optimizer' | 'other'

    @property
    def mb(self) -> float:
        return self.nbytes / MB


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_WEIGHT_SUFFIXES = (
    ".weight", ".bias", ".scale",
    ".W", ".b", ".W_in", ".W_out", ".W_Q", ".W_K", ".W_V", ".W_O",
    ".W_E", ".W_U", ".W_pos",
)
_GRAD_PREFIXES = ("grad_", "_grad", "gradient_")
_OPT_PREFIXES  = ("adam_", "adamw_", "sgd_", "optimizer.", "opt_", "m_", "v_")


def _classify(name: str) -> str:
    low = name.lower()
    if any(low.startswith(p) for p in _GRAD_PREFIXES) or ".grad" in low or low.endswith("_grad"):
        return "gradient"
    if any(low.startswith(p) for p in _OPT_PREFIXES) or ".exp_avg" in low or ".exp_avg_sq" in low:
        return "optimizer"
    if any(low.endswith(s.lower()) for s in _WEIGHT_SUFFIXES):
        return "weight"
    return "activation"


_LAYER_RE = re.compile(
    r"(?:^|[._])(?:blocks|layers|h|layer)[._]?(\d+)(?:[._]|$)"
)


def _layer_of(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class CacheMemoryProfiler:
    """
    Report memory per layer / per tensor for a HookedTransformer-style cache.
    Optionally also walks `model` to capture weights, grads, and optimizer state.
    """

    def __init__(self, cache, n_layers: int, model=None):
        self.raw_cache = cache
        self.n_layers = n_layers
        self.model = model
        self.entries: List[TensorEntry] = []
        self._collect()

# -------- cache extraction -------------------------------------------
    def _cache_tensors(self) -> Dict[str, torch.Tensor]:
        c = self.raw_cache

        # unwrap common wrappers
        if hasattr(c, "cache") and not isinstance(c, dict):
            c = c.cache
        elif hasattr(c, "caching") and not isinstance(c, dict):
            c = c.caching

        out: Dict[str, torch.Tensor] = {}

        if isinstance(c, dict):
            for k, v in c.items():
                prefix = f"{k}." if k is not None else ""
                for name, t in _iter_tensors(v, prefix):
                    key = name if name else str(k)
                    # dedupe collisions by suffixing index
                    if key in out:
                        i = 1
                        while f"{key}#{i}" in out:
                            i += 1
                        key = f"{key}#{i}"
                    out[key] = t
            return out

        if isinstance(c, (list, tuple)):
            for i, v in enumerate(c):
                for name, t in _iter_tensors(v, f"layer{i}."):
                    if name in out:
                        j = 1
                        while f"{name}#{j}" in out:
                            j += 1
                        name = f"{name}#{j}"
                    out[name] = t
            return out

        if isinstance(c, torch.Tensor):
            return {"cache": c}

        # last-ditch: try to walk it as a generic object
        found = dict(_iter_tensors(c))
        if found:
            return found

        raise TypeError(
            f"Unsupported cache type {type(c)} — expected dict/list/Tensor "
            f"or an object exposing .cache/.caching"
        )

    def _collect(self):
        for name, t in self._cache_tensors().items():
            self.entries.append(TensorEntry(
                name=name,
                shape=tuple(t.shape),
                dtype=t.dtype,
                nbytes=t.numel() * t.element_size(),
                kind=_classify(name),
            ))
        if self.model is not None:
            self._collect_model()

    # -------- model walk (weights / grads / optimizer) -------------------
    def _get_optimizer(self):
        for attr in ("optimizer", "_optimizer"):
            opt = getattr(self.model, attr, None)
            if opt is not None and hasattr(opt, "state"):
                return opt
        cfg = getattr(self.model, "cfg", None)
        if cfg is not None:
            for attr in ("optimizer", "_optimizer"):
                opt = getattr(cfg, attr, None)
                if opt is not None and hasattr(opt, "state"):
                    return opt
        trainer = getattr(self.model, "trainer", None)
        if trainer is not None:
            opt = getattr(trainer, "optimizer", None)
            if opt is not None and hasattr(opt, "state"):
                return opt
        return None

    def _collect_model(self):
        seen = {e.name for e in self.entries}

        for n, p in self.model.named_parameters():
            if n not in seen:
                self.entries.append(TensorEntry(
                    name=n, shape=tuple(p.shape), dtype=p.dtype,
                    nbytes=p.numel() * p.element_size(), kind="weight",
                ))
                seen.add(n)
            if p.grad is not None:
                gname = f"grad_{n}"
                if gname not in seen:
                    self.entries.append(TensorEntry(
                        name=gname, shape=tuple(p.grad.shape), dtype=p.grad.dtype,
                        nbytes=p.grad.numel() * p.grad.element_size(),
                        kind="gradient",
                    ))
                    seen.add(gname)

        opt = self._get_optimizer()
        if opt is not None and hasattr(opt, "state"):
            for pname, p in self.model.named_parameters():
                st = opt.state.get(p, {}) if hasattr(opt, "state") else {}
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        oname = f"opt.{pname}.{k}"
                        if oname not in seen:
                            self.entries.append(TensorEntry(
                                name=oname, shape=tuple(v.shape), dtype=v.dtype,
                                nbytes=v.numel() * v.element_size(),
                                kind="optimizer",
                            ))
                            seen.add(oname)

# -------- filtering ---------------------------------------------------
    def _filtered(
        self,
        kind: Optional[str] = None,
        layer: Optional[Union[int, List[int], tuple, set]] = None,
        include_global: bool = False,
    ) -> List[TensorEntry]:
        if layer is None:
            layers = None
        elif isinstance(layer, int):
            layers = {layer}
        else:
            layers = set(layer)

        out = []
        for e in self.entries:
            if kind is not None and e.kind != kind:
                continue
            li = _layer_of(e.name)
            if layers is not None:
                if li is None:
                    if not include_global:
                        continue
                elif li not in layers:
                    continue
            out.append(e)
        return out

    # -------- per-layer aggregation --------------------------------------
    def per_layer(self) -> Dict[int, Dict[str, float]]:
        """layer (-1 for global) -> {kind -> MB}"""
        out: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for e in self.entries:
            li = _layer_of(e.name)
            key = li if li is not None else -1
            out[key][e.kind] += e.mb
        return dict(out)

    # -------- reporting ---------------------------------------------------
    def print_table(
        self,
        kind: Optional[str] = None,
        layer: Optional[Union[int, List[int]]] = None,
        include_global: bool = False,
        sort_by_size: bool = True,
    ):
        rows = self._filtered(kind=kind, layer=layer, include_global=include_global)
        if sort_by_size:
            rows.sort(key=lambda e: -e.mb)

        hdr = f"{'name':<55} {'kind':<10} {'shape':<22} {'dtype':<10} {'MB':>8}"
        print(hdr)
        print("-" * len(hdr))
        total = 0.0
        for e in rows:
            shape = "x".join(str(s) for s in e.shape)
            dtype = str(e.dtype).replace("torch.", "")
            print(f"{e.name:<55} {e.kind:<10} {shape:<22} {dtype:<10} {e.mb:8.3f}")
            total += e.mb
        print("-" * len(hdr))
        print(f"{'TOTAL':<55} {'':<10} {'':<22} {'':<10} {total:8.3f}")

    def print_layer_table(self):
        pl = self.per_layer()
        kinds = ["weight", "activation", "gradient", "optimizer"]
        hdr = f"{'layer':>6} " + " ".join(f"{k:>12}" for k in kinds) + f" {'total MB':>12}"
        print(hdr)
        print("-" * len(hdr))
        for li in sorted(pl):
            label = "global" if li == -1 else str(li)
            cols = " ".join(f"{pl[li].get(k, 0.0):12.3f}" for k in kinds)
            print(f"{label:>6} {cols} {sum(pl[li].values()):12.3f}")

    def summary(self) -> Dict[str, float]:
        s: Dict[str, float] = defaultdict(float)
        for e in self.entries:
            s[e.kind] += e.mb
        s["total"] = sum(s.values())
        return dict(s)

    # -------- plots -------------------------------------------------------
    def plot_histograms(
        self,
        bins: int = 40,
        layer: Optional[Union[int, List[int]]] = None,
        include_global: bool = False,
        per_layer_activations: Optional[bool] = None,
    ):

        import matplotlib.pyplot as plt

        entries = (self._filtered(layer=layer, include_global=include_global)
                   if layer is not None else self.entries)

        kinds = ["weight", "activation", "gradient", "optimizer"]

        pl = self.per_layer()
        layer_ids = sorted(li for li in pl if li >= 0)
        if layer_ids:
            fig, ax = plt.subplots(figsize=(14, 4))
            bottom = np.zeros(len(layer_ids))
            for k in kinds:
                vals = np.array([pl[li].get(k, 0.0) for li in layer_ids])
                ax.bar(layer_ids, vals, bottom=bottom, label=k)
                bottom += vals
            ax.set_xlabel("layer")
            ax.set_ylabel("MB")
            ax.set_title("Per-layer memory (stacked by kind)")
            ax.legend()
            plt.show()
