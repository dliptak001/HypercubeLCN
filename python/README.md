# Hypercube LCN

[![Build wheels](https://github.com/dliptak001/HypercubeLCN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeLCN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-lcn)](https://pypi.org/project/hypercube-lcn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-lcn)](https://pypi.org/project/hypercube-lcn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeLCN/blob/main/LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

This package is the **Python** surface for HypercubeLCN
(`import hypercube_lcn`).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeLCN](https://github.com/dliptak001/HypercubeLCN)**.

HypercubeLCN is a **Locally Connected Network** on a Boolean
hypercube: a deep feedforward network whose connectivity is the
cube's own edges and whose weights are trained. It is built from two
core classes.

The **Core** class is the network. It owns the weights and runs the
forward pass: one field in, one field out, with a stack of
intermediate fields written on the same cube in between.

The **Training** class walks that same pass in reverse. It
accumulates gradients through every depth and steps the weights with
Adam.

That is the whole architecture. There is no preprocessor, no
reservoir, no separate readout — the cube is the model. In Python the
two are wrapped by a single class, **`hypercube_lcn.LCN`**.

This is the opposite bet from the sibling projects. HypercubeEtalon,
HypercubeWTF, and HypercubeCascade all put a *frozen random*
hypercube stage in front of a small trained readout, and the point of
those experiments is how far fixed dynamics can carry a thin
classifier. This project asks instead: how much better does the
hypercube do when every weight in it is trained?

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeCascade"><strong>HypercubeCascade</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeEtalon"><strong>HypercubeEtalon</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeLCN"><strong>HypercubeLCN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWorldModel"><strong>HypercubeWorldModel</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
</p>

<p align="center">
  📄 Foundational paper:
  <a href="https://github.com/dliptak001/HypercubeLCN/blob/main/docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
  (D.&nbsp;C.&nbsp;Liptak, 2026)
</p>

HypercubeLCN is an experiment in the **HypercubeAI** project — our
quest to systematically re-implement classical neural architectures
on a Boolean hypercube topology instead of Euclidean grids or random
graphs. The central thesis is "topology-native intelligence": the
hypercube's algebraic structure (vertex-transitive symmetry, Hamming
geometry, bitwise addressing) can serve as a first-class
computational substrate.

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the
  whole reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.

Each product in the family is a different architecture on that same
foundation.

---

## The LCN

A **locally connected network** is wired like a convolutional layer
— each unit reads only a small neighborhood — but where a CNN slides
one shared kernel across every position, an LCN lets every position
train its own private weights.

Here the hypercube supplies the neighborhoods. A vertex's neighbors
are the indices one bit-flip away, and at every depth each vertex
gathers from all of them at once. Each vertex owns a private weight
table at every depth — one weight for each neighbor and each field
that neighbor shows it — and shares nothing with any other vertex.
The one wrinkle is a lookback window: when a vertex reads from a
neighbor, it sees not just that neighbor's newest field but the last
few written (a configurable width, two to six). This acts as a short
skip connection and keeps the input visible to the early depths.

Training runs the forward pass's loops in reverse, and no depth is
spared: the gradient reaches every weight at every depth. For the full story,
[docs/forward.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/forward.md) and
[docs/training.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/training.md)
walk through the code loop by loop, with dim-4 examples small enough
to check by hand.

---

## Raman baseline extraction (a vibrational spectroscopy application)

The benchmark is the one the sibling projects established: recover
the slow fluorescence background under sharp molecular peaks without
lifting the baseline into the bands or cutting trenches beneath them.
The dataset is 10,000 synthetic LiCoO₂ (lithium cobalt oxide)
training spectra and 2,000 held-out validation spectra, scored as
RMSE in raw counts.

On this task the frozen-stage siblings — Etalon, WTF, and Cascade,
each feeding the same one-layer, one-channel readout — all landed on
one floor: 4.76 to 4.82 validation. The LCN, with a dim-11 cube
matched to the 2048-bin spectrum and every weight trained, scores
**1.98 training / 2.03 validation**.

![Held-out validation extract, spectra 351 through 354](https://raw.githubusercontent.com/dliptak001/HypercubeLCN/main/examples/RamanBaseline/extracted_baselines.png)

Grey is the raw spectrum, red the true baseline, blue the extract. At
two counts of RMSE the residual is at the scale of the label's own
noise, and the red trace all but disappears under the blue. A second
experiment widens the cube to dim 12 so half the vertices serve as
free hidden units, and scores **1.64 / 1.69**. The write-ups are
[examples/RamanBaseline/](https://github.com/dliptak001/HypercubeLCN/blob/main/examples/RamanBaseline/README.md) and
[examples/RamanBaselineNarrowIO/](https://github.com/dliptak001/HypercubeLCN/blob/main/examples/RamanBaselineNarrowIO/README.md).

---

## Installation

**Preferred:** install a pre-built wheel from PyPI (no compiler).

```bash
pip install hypercube-lcn
```

```python
import hypercube_lcn as hl
print(hl.__version__)
```

Package name on PyPI: **`hypercube-lcn`**. Import name:
**`hypercube_lcn`**. Main type: **`hl.LCN`**.

Wheels target Python 3.10–3.14 on common Windows, Linux, and macOS machines.
Runtime dependency: NumPy only.

### From source (full repository)

To compile the extension yourself, clone this **entire** repository (not a
minimal source-only download of the `python/` folder alone — the C++ core
lives next to `python/`). You need Python 3.10+, a C++23 compiler, and
CMake ≥ 3.20.

```bash
git clone https://github.com/dliptak001/HypercubeLCN.git
cd HypercubeLCN/python
pip install .
```

On Windows with CLion's MinGW, put that compiler's `bin` folder (and Ninja) on
your `PATH`, then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

(Exact CLion paths change with the version.)

---

## Quick start

You bring each sample as a length-**N** float array (N = 2<sup>dim</sup>). How
you get there — pad an image, reshape a spectrum, invent a layout — is up to
you. This package does not pack 784 pixels or 2048 bins for you.

Shapes that matter:

| Array | Shape | Notes |
|-------|-------|-------|
| `fields` | `(count, N)` | one length-N field per row |
| `targets` | `(count, width)` | width ≤ N; width < N masks the loss to vertices 0..width-1 |

```python
import numpy as np
import hypercube_lcn as hl

dim = 6
N = 2**dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((256, N)).astype(np.float32)
targets = 0.5 * (fields + fields[:, np.arange(N) ^ 1])  # a local map

net = hl.LCN(dim=dim, gather_span=3, tanh_last=False,
             lr=1e-2, lr_min_frac=0.05, restore_best=True)
net.fit(fields, targets, epochs=40, batch_size=16, verbose=True)

prediction = net.forward(fields[0])   # (N,) float32
print(net)

net.save("model.pkl")
loaded = hl.LCN.load("model.pkl")
```

### Step by step (same loop, more control)

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
    net.observe(epoch_metric, epoch)      # restore_best bookkeeping
net.restore_best()
```

The gradient is a **sum** over the batch, so the effective step scales with
batch size — the same convention as the C++ examples.

---

## Features

- **One class** — `hypercube_lcn.LCN` is the whole product surface
- **`fit`** — shuffle, batch, cosine schedule, restore-best, in one call
- **Custom loops** — `zero_grad` / `forward` / `loss` / `backward` / `adam` /
  `set_epoch` exposed one-to-one with the C++ API
- **dim 4–24** — field length N = 2<sup>dim</sup>; depth `z_max`
  (default: dim); lookback window `gather_span` 2–6
- **Masked loss** — targets narrower than N leave the rest of the cube as
  free hidden units
- **`tanh_last`** — off by default (raw accumulator out); on confines the
  output to (-1, 1)
- **Weights and gradient as NumPy** — `net.weights` (settable) and
  `net.grad`, z-major layout: depth, vertex, axis, tap
- **Save / load** — `save` / `load` (pickle: config + weights; optimizer
  state is not stored)
- **NumPy float32** — arrays converted for you; prefer contiguous float32

---

## Examples

For a first try, paste the [Quick start](#quick-start) after
`pip install hypercube-lcn`. That is self-contained.

The demo scripts on GitHub under
[`python/examples/`](https://github.com/dliptak001/HypercubeLCN/tree/main/python/examples)
are there to open or download — they are not added to your machine by pip.

| Script | What it is for |
|--------|----------------|
| [synthetic_regression.py](https://github.com/dliptak001/HypercubeLCN/blob/main/python/examples/synthetic_regression.py) | Toy field map: `fit` with verbose loss, then held-out MSE |

```bash
# from a clone of HypercubeLCN, after: pip install hypercube-lcn
python python/examples/synthetic_regression.py
```

These use easy made-up fields so the API is obvious — not scores to publish.

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/Python_SDK.md)** | Canonical Python API — every method, layout, pickle, limits |
| [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/CPP_SDK.md) | Native library guide (same product, C++) |
| [Project README](https://github.com/dliptak001/HypercubeLCN#readme) | Product story and C++ demos from the repo root |
| [docs/forward.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/forward.md) | The forward pass, loop by loop, with hand-checkable examples |
| [docs/training.md](https://github.com/dliptak001/HypercubeLCN/blob/main/docs/training.md) | Backprop and Adam through the same loops |
| [examples/README.md](https://github.com/dliptak001/HypercubeLCN/blob/main/examples/README.md) | The C++ example programs and datasets |

---

## Ecosystem

- **[HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)**: both frozen stages in series + thin readout.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)**: cube-native conv stack with shared kernels.
- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)**: echo-state / reservoir computing on streams.
- **[HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon)**: frozen etalon transit + thin readout.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)**: Hopfield-style dynamics on the cube.
- **[HypercubeLCN](https://github.com/dliptak001/HypercubeLCN)**: the locally connected net, every weight trained.
- **[HypercubeWorldModel](https://github.com/dliptak001/HypercubeWorldModel)**: frozen WTF encoder + trained LCN predictor and decoder.
- **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)**: frozen reservoir orbit + thin readout.

---

## License

Apache 2.0.
