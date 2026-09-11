# HypercubeLCN Python SDK

HypercubeLCN is a **locally connected network on a Boolean hypercube** — a
deep feedforward net whose connectivity is the cube's own edges and whose
weights are **all trained**. One class — `LCN` — owns the whole product:
forward, fit, custom training loops, weights, persistence.

This is a **map API**, not a stream API: each sample is one full length-N
field, one forward pass, one output field. There is no per-tick input
sequence (that is
[HypercubeESN](https://github.com/dliptak001/HypercubeESN)).

C++ core and contracts: **[CPP_SDK.md](CPP_SDK.md)**.  
The passes, loop by loop: **[forward.md](forward.md)** / **[training.md](training.md)**.  
PyPI-facing package story: **[python/README.md](../python/README.md)**.  
Package version: single source `python/hypercube_lcn/_version.py`
(`hypercube_lcn.__version__` and wheel metadata both read it).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What a pass is](#what-a-pass-is)
- [Vocabulary](#vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Data types](#data-types)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Installation

### From PyPI (preferred)

Pre-built **wheels** — no compiler required:

```bash
pip install hypercube-lcn
```

Import as `import hypercube_lcn as hl` (PyPI name `hypercube-lcn`).
Wheels cover Python 3.10–3.14 on common Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64) builds. NumPy is the only runtime
dependency.

### From source (full repository)

Compile only from a **full clone** of HypercubeLCN. The extension compiles
the C++ core (`Core.cpp`, `Training.cpp`) that sits **outside** the `python/`
package directory; a `python/`-only tree is not enough.

Requirements: Python 3.10+, C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+),
CMake 3.20+, scikit-build-core, pybind11, NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeLCN.git
cd HypercubeLCN/python
pip install .
```

On Windows with MinGW (e.g. CLion toolchain):

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

From the `python/` directory after install:

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

Or from the repository root: `pytest python/tests/ -v --import-mode=importlib`.
Importlib mode avoids the source tree shadowing the installed `_core`
extension. Use the `pytest` entry point, not `python -m pytest` — the latter
puts the current directory on `sys.path`, and from `python/` the source
package (which has no compiled `_core`) then shadows the installed one.

### Examples

The [Quick start](#quick-start) below is enough after `pip install`. Longer
demos live in the **git tree** under
[`python/examples/`](../python/examples/README.md) — they are **not** part of
the wheel. From a clone, repository root:

```bash
pip install hypercube-lcn   # or: pip install ./python
python python/examples/synthetic_regression.py
```

## Quick start

```python
import numpy as np
import hypercube_lcn as hl

dim = 6
N = 1 << dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((256, N)).astype(np.float32)
targets = (0.5 * (fields + fields[:, np.arange(N) ^ 1])).astype(np.float32)

net = hl.LCN(dim=dim, gather_span=3, tanh_last=False,
             lr=1e-2, lr_min_frac=0.05, restore_best=True)
net.fit(fields, targets, epochs=40, batch_size=16, verbose=True)

prediction = net.forward(fields[0])   # (N,) float32
```

### Explicit (full control)

`fit` is nothing but this loop — drive it yourself to interleave your own
metrics, schedules, or early stopping:

```python
for epoch in range(epochs):
    net.set_epoch(epoch, epochs)          # cosine learning rate
    for start in range(0, count, batch_size):
        net.zero_grad()
        for i in range(start, start + batch_size):
            net.forward(fields[i])
            net.loss(targets[i])          # seeds the gradient
            net.backward()                # accumulates (sums) it
        net.adam()                        # one step per batch
    net.observe(val_metric, epoch)        # restore_best: lower wins
net.restore_best()
```

The order matters and nothing enforces it for you: `loss` reads the most
recent `forward`, `backward` reads what `forward` and `loss` left behind, and
the gradient is a **sum** over the batch — the effective step scales with
batch size.

### Data-parallel (threads)

The gradient is a sum, so a batch splits across plain `threading.Thread`
workers: the extension releases the GIL during `forward` / `loss` /
`backward`, so separate instances on separate threads get real
parallelism. Each worker is a clone (same constructor knobs); one master
merges with `accumulate_grad` and takes the single `adam` step:

```python
workers = [hl.LCN(dim=net.dim, z_max=net.z_max,
                  gather_span=net.gather_span) for _ in range(T)]

for start in range(0, count, batch_size):
    for w in workers:
        w.weights = net.weights           # sync to the master
    def work(w, lo):                      # strided share of the batch
        w.zero_grad()
        for i in range(start + lo, start + batch_size, T):
            w.forward(fields[i]); w.loss(targets[i]); w.backward()
    threads = [threading.Thread(target=work, args=(w, k))
               for k, w in enumerate(workers)]
    for t in threads: t.start()
    for t in threads: t.join()
    net.zero_grad()
    for w in workers:                     # fixed order: deterministic
        net.accumulate_grad(w.grad)       # for a given T
    net.adam()
```

Adam moments, `observe`, and `restore_best` stay on the master. Results
are deterministic for a fixed `T` but differ from a serial run in float
summation order. (The C++ MNIST and Raman examples run this same
pattern natively.)

## What a pass is

```text
x  (length-N field, host-packed)
    │
    ▼
 history stack: [zeros … zeros | x]
    │
    ▼  for each depth (z_max of them):
 every vertex gathers its dim bit-flip neighbors × gather_span fields,
 weighs each value with its own private table, tanh, writes the next field
    │
    ▼
 last field written  =  output (length N)
```

- **N = 2^dim** vertices / field length (dim 4…24).
- Every vertex owns a private weight table at every depth — nothing is
  shared, and **everything trains**: the gradient reaches all
  N × dim × gather_span × z_max weights.
- Host packing (images → N, spectra → N, …) is the host's job — this
  package does not reshape domain data onto the cube.
- Targets may be **narrower than N**: then only vertices `0 .. width-1`
  carry loss and the rest of the cube is free hidden units.

## Vocabulary

| Term | Meaning |
|------|---------|
| **Field** | Length-N float32 vector on the cube (you pack domain data) |
| **Depth** | One gather-and-write over all vertices; `z_max` of them per pass |
| **Lookback window** | Each read sees a neighbor's last `gather_span` fields, not just the newest |
| **Masked loss** | A target narrower than N constrains only vertices 0..width-1 |
| **Free hidden vertices** | The unconstrained rest of the cube — computation without loss |
| **N** | Vertices / field length = 2^dim |
| **z_max** | Depth count; constructor 0 means "use dim" |
| **gather_span** | Lookback window width, 2…6 |
| **tanh_last** | False = last depth writes the raw accumulator (raw-unit regression) |

## API reference

### Constructor `LCN(dim, **kwargs)`

All knobs are fixed at construction (same contract as the C++ `CoreConfig` +
`TrainingConfig` pair).

```python
import hypercube_lcn as hl

net = hl.LCN(
    dim=11,                    # required; 4-24; N = 2^dim
    seed=934791766227647176,   # weight-init seed
    z_max=0,                   # depths; 0 = use dim, else >= 2
    gather_span=2,             # lookback window width, 2-6
    tanh_last=False,           # True: confine the output to (-1, 1)
    lr=5e-3,                   # Adam step; cosine peak (finite, > 0)
    lr_min_frac=1.0,           # cosine floor as a fraction of lr; [0, 1]
    lr_decay_epochs=0,         # cosine horizon; 0 = fit's epochs
    restore_best=False,        # snapshot best weights during fit
    beta1=0.9,                 # Adam first-moment decay [0, 1)
    beta2=0.999,               # Adam second-moment decay [0, 1)
    eps=1e-8,                  # Adam denominator floor (finite, > 0)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | required | Hypercube dimension **[4, 24]**. N = 2^dim. |
| `seed` | `int` | `934791766227647176` | Weight-init seed (normal draw, Xavier-style scale). |
| `z_max` | `int` | `0` | Depth count; **0 = use dim**, else must be ≥ 2. Depth buys reach: z depths let vertices z bits apart interact. |
| `gather_span` | `int` | `2` | Lookback window width, **[2, 6]**. The Raman runs use 4. |
| `tanh_last` | `bool` | `False` | Default: the last depth writes the raw accumulator (what squared-error training wants). `True` confines the output to (-1, 1). |
| `lr` | `float` | `5e-3` | Adam step size and cosine peak. Finite, > 0. |
| `lr_min_frac` | `float` | `1.0` | Cosine floor = `lr * lr_min_frac`; **[0, 1]**; 1 = constant lr. |
| `lr_decay_epochs` | `int` | `0` | Cosine horizon; 0 = use `fit`'s `epochs`. ≥ 0. |
| `restore_best` | `bool` | `False` | Snapshot weights on a new low observed metric; `fit` restores at the end. |
| `beta1` | `float` | `0.9` | Adam first-moment decay, **[0, 1)**. |
| `beta2` | `float` | `0.999` | Adam second-moment decay, **[0, 1)**. |
| `eps` | `float` | `1e-8` | Adam denominator floor. Finite, > 0. |

**Sizing note:** the weight count is `N × dim × gather_span × z_max` — it
grows fast with dim. The dim-12 Raman configuration is about 1.6 M weights;
at the top of the dim range allocation itself is the limit.

### Methods

| Method | Role |
|--------|------|
| `forward(x)` | One forward pass; returns the length-N float32 output field (a copy). |
| `fit(fields, targets, *, epochs, batch_size=32, shuffle_seed=1, verbose=False)` | Shuffle, batch, cosine schedule, restore-best — the standard cycle. Returns `self`. Calling again continues from the current weights. |
| `zero_grad()` | Clear the accumulated gradient (start of each batch). |
| `loss(target)` | 0.5 × SSE against the most recent `forward`; seeds the gradient. `len(target) == N` is full width; shorter is masked to vertices 0..len-1. |
| `backward()` | Backprop the most recent `loss`; **accumulates (sums)** into the gradient. Once per sample. |
| `adam()` | One Adam step on the accumulated gradient. |
| `accumulate_grad(g)` | Add an external gradient (a worker's `grad`) into this network's gradient — the data-parallel merge. Exact length required. |
| `set_epoch(epoch, num_epochs=0)` | Apply the cosine schedule for this epoch. |
| `observe(metric, epoch)` | restore-best bookkeeping; **lower wins, strictly**. |
| `restore_best()` | Write the best-metric snapshot back into the network. |
| `reset()` | Forget the optimizer run: Adam moments, step count, lr, best snapshot. Weights untouched. |
| `save(path)` / `LCN.load(path)` | Pickle constructor config + weights. |

### Properties

| Property | Meaning |
|----------|---------|
| `dim`, `N` | Geometry (N = 2^dim) |
| `z_max` | Resolved depth count (constructor 0 already replaced by dim) |
| `gather_span`, `tanh_last`, `seed` | Config mirrors |
| `num_weights` | N × dim × gather_span × z_max |
| `weights` | All weights as a float32 array (z-major: depth, vertex, axis, tap). **Settable** — exact length required |
| `grad` | The accumulated gradient, same layout as `weights` |
| `lr` | Learning rate currently in effect (after `set_epoch`) |
| `has_best`, `best_epoch`, `best_metric` | restore-best state |

## Input data layout

- **Fields** must be length **N** per sample. `fit` takes shape `(count, N)`;
  single-sample methods accept any array that ravel-flattens to N.
- **Targets** are shape `(count, width)` with **width ≤ N**. Width == N is
  full-field regression; smaller widths mask the loss to vertices
  `0 .. width-1` — pack your task there.
- **Classification** is not a separate mode: encode classes as a one-hot
  ±1 field. Either mask the loss to the first `num_classes` vertices and
  argmax those outputs, or stripe the pattern across all N outputs
  (vertex `v` targets class `v % num_classes`) and argmax the per-class
  means (the C++ MNIST demo is the worked recipe for the striped form).
- **Host packing** (images, spectra, sensors → N) is outside this package.

## Data types

| Role | Preferred type | Notes |
|------|----------------|-------|
| Fields / targets / weights / predictions | `float32` | Other dtypes converted via NumPy to contiguous float32 |
| `epochs`, `batch_size`, epochs in `set_epoch` / `observe` | `int` | |

## Error handling

Python-side checks raise `ValueError` with a short message (bad `dim`, field
or target shape, weight length, `epochs`/`batch_size` < 1). Native
`std::invalid_argument` — the C++ constructors validate every config range,
and `forward`/`loss` validate lengths — maps to `ValueError`; other C++
failures surface as `RuntimeError` via pybind11.

Typical mistakes:

- Field length ≠ N
- Target wider than N (masked loss is a width ≤ N, not a mask vector)
- `gather_span` outside [2, 6], or `z_max = 1` (0 means "use dim"; otherwise ≥ 2)
- `beta1` / `beta2` at 1.0, or a non-positive `lr` / `eps`
- Loading a weight array whose length does not match `num_weights`
- `backward()` without a fresh `loss()` against the most recent `forward()` —
  the stale-gradient guard raises `ValueError`. Each `loss()` seeds one
  `backward()`, and `adam()` or a weights load invalidates a pending seed
- Forgetting `zero_grad` between batches — the gradient keeps summing
- Feeding `observe` an accuracy (it wants lower-is-better)

## Model persistence

| Mechanism | What is stored | Optimizer state? |
|-----------|----------------|------------------|
| `save` / `pickle` | Constructor config + all weights | **No** (Adam moments, step count, best snapshot are not saved) |

Pickle version is bumped when the serialized layout changes; newer libraries
reject unknown future versions with an upgrade message.

```python
net.save("model.pkl")
net2 = hl.LCN.load("model.pkl")   # same ctor knobs + weights; fresh optimizer
```

A pickle captures the whole product: the network reconstructs from the
constructor knobs and the weight array is loaded verbatim. `forward` on the
loaded instance reproduces the original bit for bit. To continue training a
loaded model, just call `fit` — but know that Adam starts cold.

The raw weight array is also yours through `net.weights` (settable), so any
external format — `np.save`, HDF5, a database — works without pickle.

**Security:** `load` uses `pickle.load`. Never load untrusted files.

## Limitations

- One `LCN` instance is **not thread-safe** for concurrent calls from
  multiple host threads. Separate instances on separate threads are fine —
  the extension releases the GIL during `forward` / `loss` / `backward` /
  `adam`, so multi-instance threading gets real parallelism (see the
  data-parallel recipe above; `accumulate_grad` is the merge).
- `fit` computes no held-out metric and observes the **train** loss for
  restore-best. For validation-driven restore-best, drive the explicit loop
  and feed `observe` your own metric.
- `fit` drops a short tail each epoch so every Adam step sees exactly
  `batch_size` samples (unless the whole set is smaller than one batch).
- There is no built-in noise, augmentation, or packing helper — hosts add
  those around the loop.
- Native contracts and the C++ surface: **[CPP_SDK.md](CPP_SDK.md)**.

## Dependencies

| Layer | What |
|-------|------|
| Runtime | NumPy |
| Wheel install | No compiler |
| From-source build | Full repo clone, C++23, CMake ≥ 3.20, scikit-build-core, pybind11 |

The C++ core is compiled into the extension itself.
