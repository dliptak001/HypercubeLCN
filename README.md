# Hypercube LCN

[![Build wheels](https://github.com/dliptak001/HypercubeLCN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeLCN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-lcn)](https://pypi.org/project/hypercube-lcn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-lcn)](https://pypi.org/project/hypercube-lcn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

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
reservoir, no separate readout — the cube is the model.

This is the opposite bet from the sibling projects. HypercubeEtalon,
HypercubeWTF, and HypercubeCascade all put a *frozen random*
hypercube stage in front of a small trained readout, and the point of
those experiments is how far fixed dynamics can carry a thin
classifier. This project asks instead: can the hypercube do better
when every weight in it is trained?

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
  <a href="docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
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

The forward pass goes something like this.

    Start the history stack with enough fields of zeros to fill the
    lookback window, then copy the input field in after them. (The
    zeros are padding: the earliest depths want to read further back
    than the input, and what they find there is zeros.)

    For each depth:

        Every vertex xors its way to each of its neighbors, reads
        that neighbor's fields across the lookback window, and weighs
        every value with its own private table.

        tanh the sum. Write it as the next field. (The last depth
        may skip the tanh — regression targets are not confined
        to (-1, 1).)

    The last field written is the output.

Training runs the same loops in reverse, and no depth is spared: the
gradient reaches every weight at every depth. For the full story,
[`docs/forward.md`](docs/forward.md) and
[`docs/training.md`](docs/training.md) walk through the code loop by
loop, with dim-4 examples small enough to check by hand.

---

## Raman baseline extraction (a vibrational spectroscopy application)

The benchmark is the one the sibling projects established: recover
the slow fluorescence background under sharp molecular peaks without
lifting the baseline into the bands or cutting trenches beneath them.
Polynomials, asymmetric least squares, and ordinary convolutional
nets follow the empty stretches well and then fail where it matters.
The dataset is 10,000 synthetic LiCoO₂ (lithium cobalt oxide)
training spectra and 2,000 held-out validation spectra, scored as
RMSE in raw counts.

On this task the frozen-stage siblings — Etalon, WTF, and Cascade,
each feeding the same one-layer, one-channel readout — all landed on
one floor: 4.76 to 4.82 validation, three preprocessors with no
shared mechanism separated by six hundredths of a count.

The LCN, with a dim-11 cube matched to the 2048-bin spectrum and
every weight trained, scores **1.98 training / 2.03 validation**.

![Held-out validation extract, spectra 351 through 354](examples/RamanBaseline/extracted_baselines.png)

Grey is the raw spectrum, red the true baseline, blue the extract. At
two counts of RMSE the residual is at the scale of the label's own
noise, and the red trace all but disappears under the blue — through
peak clusters, along noisy troughs, down steep flanks, on all 2,000
held-out spectra. The full write-up, knobs, and scoring definition
are in [`examples/RamanBaseline/`](examples/RamanBaseline/README.md).

### Narrow IO: free hidden vertices

The dim-11 run has a peculiar symmetry: input, output, and every unit
of computation share the same 2048 vertices. A second experiment
evaluates what happens when that symmetry is broken. Keep the
2048-wide IO but run a dim-12 cube: the spectrum occupies half the
vertices, and the other 2048 are fed zeros and owe nothing to the
loss — free hidden units, the network's first.

Result: **1.64 training / 1.69 validation** — 17 % below the dim-11
baseline, with the same 0.05-count train/validation gap. The free
half of the cube earns its keep, and the door opens to tasks whose
natural width is not a power of two. The write-up is
[`examples/RamanBaselineNarrowIO/`](examples/RamanBaselineNarrowIO/README.md).

The 1,572,864 trainable weights sound alarming next to 10,000
training spectra, but consider that each spectrum constrains all
2048 output bins — some 20 million scalar constraints in total — and
both runs hold a train/validation gap of only five hundredths of a
count.

---

## Using it

The whole library is two header files, [`Core.h`](Core.h) and
[`Training.h`](Training.h), written in plain C++ with no dependencies
beyond the standard library. Include them and you have the network
and its trainer.

The easiest way in, though, is through the worked examples in
[`examples/`](examples/README.md). They cover downloading the
dataset, building the model, and running the training loop
end to end. There are no command-line arguments anywhere — every
knob is a `constexpr` at the top of the example's source file, so
configuring a run means editing a few lines and rebuilding.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
