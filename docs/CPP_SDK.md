# HypercubeLCN C++ SDK

HypercubeLCN is a **locally connected network on a Boolean hypercube** — a
deep feedforward net whose connectivity is the cube's own edges and whose
weights are **all trained**. Two classes own the whole product: `Core` runs
the forward pass, `Training` runs it backwards and steps the weights with
Adam. Both live in two headers at the repository root, written in plain
C++23 with no dependencies beyond the standard library.

This is a **map API**, not a stream API: each sample is one full length-N
field, one forward pass, one output field. There is no per-tick input
sequence (that is
[HypercubeESN](https://github.com/dliptak001/HypercubeESN)).

The same product from Python: **[Python_SDK.md](Python_SDK.md)**.  
The passes, loop by loop: **[forward.md](forward.md)** / **[training.md](training.md)**.  
Worked programs: **[examples/README.md](../examples/README.md)**.  
This guide matches the public headers for **1.0.x**.

## Contents

- [Build and link](#build-and-link)
- [Quick start](#quick-start)
- [What a pass is](#what-a-pass-is)
- [Vocabulary](#vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Build and link

Requirements: a **C++23** compiler (GCC 13+, Clang 17+, MSVC 2022+) and
**CMake ≥ 3.20**. The library target is **`HypercubeLCNCore`** — the two
sources `Core.cpp` / `Training.cpp` as a static library, exporting the repo
root as a public include directory.

From your own project:

```cmake
add_subdirectory(path/to/HypercubeLCN)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeLCNCore)
```

```cpp
#include "Core.h"
#include "Training.h"
```

The example executables build alongside; ignore them or exclude them from
your default target.

Building this repo directly (CLion: open, reload CMake, build — or any
shell with the toolchain available):

```bash
cmake --build cmake-build-release
```

Use **Release** for real runs. Release builds compile with
`-O3 -ffast-math` plus `-march` per the `HYPERCUBE_ARCH` cache variable
(same knob as the Python wheel): `native` by default for local builds,
`x86-64-v2` for portable release binaries, `none` to skip `-march`
(ARM, MSVC). Fast-math in Release is the family convention, shared with
HypercubeCascade.

### Example binaries

| Binary | Role |
|--------|------|
| `lcn_raman` | Raman baseline regression, dim-11 full-width |
| `lcn_raman_extract` | Writes baseline extracts from a trained run |
| `lcn_raman_narrow` | The dim-12 narrow-IO variant (free hidden vertices) |
| `lcn_raman_narrow_extract` | Extracts for the narrow-IO run |
| `lcn_mnist` | MNIST as striped one-hot regression |

Every knob in the examples is a `constexpr` at the top of a source file;
there are no command-line arguments. Data locations and run notes:
[examples/README.md](../examples/README.md).

## Quick start

A complete toy regression. Verified against the library: it compiles as
shown, and the printed loss falls from 7.19 to 0.08.

```cpp
#include "Core.h"
#include "Training.h"
#include <cstdio>
#include <random>
#include <vector>

int main() {
    CoreConfig cfg;
    cfg.dim = 5;               // N = 32
    cfg.gather_span = 3;
    cfg.tanh_last = false;     // regression targets, not confined to (-1, 1)

    TrainingConfig tcfg;
    tcfg.lr = 1e-2f;
    tcfg.lr_min_frac = 0.05f;  // cosine decay to 5 % of lr

    auto core = Core::Create(cfg);
    Training train(*core, tcfg);
    const size_t N = core->N();

    // Toy task: predict the mean of each vertex and its axis-0 neighbor.
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.f, 1.f);
    const int count = 256;
    std::vector<float> fields(count * N), targets(count * N);
    for (auto& f : fields)
        f = dist(rng);
    for (int i = 0; i < count; ++i)
        for (size_t v = 0; v < N; ++v)
            targets[i * N + v] =
                0.5f * (fields[i * N + v] + fields[i * N + (v ^ 1)]);

    const int epochs = 30, batch = 16;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        train.SetEpoch(epoch, epochs);
        float loss_sum = 0.f;
        for (int start = 0; start < count; start += batch) {
            train.ZeroGrad();
            for (int i = start; i < start + batch; ++i) {
                core->Forward(std::span<const float>{&fields[i * N], N});
                loss_sum += train.Loss(
                    std::span<const float>{&targets[i * N], N});
                train.Backward();
            }
            train.Adam();
        }
        if (epoch % 10 == 0 || epoch == epochs - 1)
            std::printf("epoch=%d mean_loss=%.5f\n",
                        epoch, loss_sum / count);
    }
    return 0;
}
```

### The cycle

Every real run is that same loop with restore-best around it:

```text
fill CoreConfig + TrainingConfig
Core::Create once, construct Training on it
for each epoch:
    SetEpoch                     (cosine learning rate)
    for each batch:
        ZeroGrad
        per sample: Forward → Loss → Backward
        Adam
    Observe                      (restore_best: lower wins)
RestoreBest
Forward                          (inference: Output() is the prediction)
```

The order matters: `Loss` reads the most recent `Forward`, `Backward` reads
what `Forward` and `Loss` left behind, and the gradient is a **sum** over
the batch — the effective Adam step scales with batch size, so pick `lr`
and batch size together. The stale-gradient guard throws on the common
out-of-order mistakes ([Error handling](#error-handling)); it cannot catch
a forgotten `ZeroGrad` or `SetEpoch`.

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
- The lookback window is a short skip connection: depths `0 .. span-1`
  still see the raw input directly; from depth `span` on, the network
  works from its own mixtures. The last depth may skip the tanh
  (`tanh_last = false`) so regression targets are not confined to (-1, 1).
- Host packing (images → N, spectra → N, …) is the host's job — the
  library does not reshape domain data onto the cube.
- Targets may be **narrower than N**: then only vertices `0 .. width-1`
  carry loss and the rest of the cube is free hidden units.

## Vocabulary

| Term | Meaning |
|------|---------|
| **Field** | Length-N float vector on the cube (you pack domain data) |
| **Depth** | One gather-and-write over all vertices; `z_max` of them per pass |
| **Lookback window** | Each read sees a neighbor's last `gather_span` fields, not just the newest |
| **Masked loss** | A target narrower than N constrains only vertices 0..width-1 |
| **Free hidden vertices** | The unconstrained rest of the cube — computation without loss |
| **N** | Vertices / field length = 2^dim |
| **z_max** | Depth count; config 0 means "use dim" |
| **gather_span** | Lookback window width, 2…6 |
| **tanh_last** | false = last depth writes the raw accumulator (raw-unit regression) |

## API reference

Authoritative signatures and contracts live in **`Core.h`** and
**`Training.h`** — both are fully doc-commented. This section is the
host-oriented map.

### Configuration

Everything is fixed at construction; both constructors validate and throw
`std::invalid_argument` on violations.

```cpp
struct CoreConfig {
    size_t dim = 8;           // hypercube dimension [4, 24]; N = 2^dim
    uint64_t seed = ...;      // weight-init seed (normal, Xavier-style scale)
    size_t z_max = 0;         // depths; 0 = use dim, else must be >= 2
    size_t gather_span = 2;   // lookback window width [2, 6]
    bool tanh_last = false;   // true: tanh on the last depth too
};

struct TrainingConfig {
    float lr = 5e-3f;           // Adam step size; cosine peak (finite, > 0)
    float lr_min_frac = 1.f;    // floor = lr * lr_min_frac; [0, 1]; 1 = constant
    int lr_decay_epochs = 0;    // cosine horizon; 0 = SetEpoch's num_epochs
    bool restore_best = false;  // snapshot weights on a new low metric
    float beta1 = 0.9f;         // Adam first-moment decay [0, 1)
    float beta2 = 0.999f;       // Adam second-moment decay [0, 1)
    float eps = 1e-8f;          // Adam denominator floor (finite, > 0)
};
```

| Knob | Guidance |
|------|----------|
| `dim` | Sized to your task width; the Raman runs use 11 and 12 |
| `z_max` | Depth buys reach: z depths let vertices z bits apart interact. Default (dim) gives antipodal reach |
| `gather_span` | Wider keeps the input visible longer; the Raman runs use 4 |
| `tanh_last` | Default off (raw accumulator); turn on only if you truly want the output confined to (-1, 1) |
| `lr`, batch size | The gradient is a batch **sum** — scale them together |
| `restore_best` | On for real runs; feed `Observe` a validation metric |

**Sizing note:** the weight count is `N × dim × gather_span × z_max` — it
grows fast with dim. The dim-12 Raman configuration is about 1.6 M weights;
at the top of the dim range allocation itself is the limit, and the
constructor's intended failure mode there is `std::bad_alloc`.

### Core — run the network

```cpp
auto core = Core::Create(cfg);        // unique_ptr; ctor validates

core->Forward(std::span<const float>{x});  // length-checked
core->Forward(x_ptr);                      // pointer form: N floats, trusted
const std::vector<float>& y = core->Output();   // length N; valid until next Forward

core->Dim();  core->N();  core->ZMax();  core->GatherSpan();
core->TanhLast();  core->Seed();
core->Config();                        // resolved CoreConfig (z_max filled in)

const std::vector<float>& w = core->Weights();  // z-major: depth, vertex, axis, tap
core->LoadWeights(std::span<const float>{w2});  // exact-length replacement
```

### Training — train it

```cpp
Training train(*core, tcfg);           // core must outlive train

train.ZeroGrad();                      // start of each batch
float e = train.Loss(std::span<const float>{target});   // after Forward;
                                       // target.size() == N: full width,
                                       // smaller: masked to 0..size-1
train.Backward();                      // once per Loss; accumulates
train.Adam();                          // one step per batch

train.SetEpoch(epoch, num_epochs);     // cosine lr; once per epoch
train.Lr();                            // rate currently in effect

train.Observe(metric, epoch);          // restore_best: lower wins, strict
train.RestoreBest();                   // write the snapshot back
train.HasBest();  train.BestEpoch();  train.BestMetric();

train.Reset();                         // forget the run: moments, t, lr, best
const std::vector<float>& g = train.Grad();   // same layout as Weights()
train.AccumulateGrad(std::span<const float>{g2});  // add an external
                                       // gradient (a worker's Grad()) —
                                       // the data-parallel merge
train.Config();
```

The data-parallel pattern: worker Trainings on Core clones each build a
private gradient over a share of the batch (one instance is not
thread-safe, but separate instances on separate threads are); the master
merges each worker's `Grad()` with `AccumulateGrad` in fixed order and
takes the single `Adam` step, then workers re-sync via `LoadWeights`.
`examples/common/worker_pool.h` plus the MNIST and Raman examples are
the worked recipe.

Pointer forms of `Loss` exist for hosts that already hold raw buffers:
`Loss(ptr)` assumes N floats; `Loss(ptr, count)` is the masked form.

`CosineLR(lr_max, lr_min, epoch, num_epochs)` — the schedule itself — is a
free function, the same formula as Etalon / Cascade / WTF.

## Input data layout

- **Fields** must be length **N**. The span overloads check; the pointer
  overloads cannot (they trust you to hand them N floats).
- **Targets** are width ≤ N. Width == N is full-field regression; smaller
  widths mask the loss to vertices `0 .. width-1` — a prefix width, not a
  mask vector — so pack your task there.
- **Classification** is not a separate mode: encode classes as a one-hot
  ±1 field. Either mask the loss to the first `num_classes` vertices and
  argmax those outputs, or stripe the pattern across all N outputs
  (vertex `v` targets class `v % num_classes`) and argmax the per-class
  means — [lcn_mnist](../examples/mnist/lcn_mnist.cpp) is the worked
  recipe for the striped form.
- **Host packing** (images, spectra, sensors → N) is outside the library.
  `examples/common/` holds optional recipes (spatial embed, augmentation).
- Everything is `float`; sizes and indices are `size_t`.

## Error handling

Constructors validate every config range and throw
`std::invalid_argument`; so do `Forward` / `Loss` / `LoadWeights` on bad
lengths, and `Backward` on an out-of-order cycle. Oversized configs throw
`std::bad_alloc` (or `std::length_error`) from allocation.

Typical mistakes:

| Symptom / assumption | Fix |
|----------------------|-----|
| Throw on `Forward` | Span length must equal `core->N()`; pack first |
| Construct throws on dim | Valid range is [4, 24] |
| Construct throws on z_max | 0 means "use dim"; any other value must be ≥ 2 |
| Construct throws on a Training knob | `lr` finite > 0, `lr_min_frac` in [0, 1], betas in [0, 1), `eps` > 0 |
| Loss looks huge | It is 0.5 × **sum** of squared error over the target width, not a mean |
| Throw on `Backward` | Take a fresh `Loss` against the most recent `Forward` first — `Forward(a)`, `Loss`, `Forward(b)`, `Backward` is the classic misuse; a second `Backward` off one `Loss`, or `Adam` / `LoadWeights` between `Loss` and `Backward`, trips the same guard |
| Loss never falls | Check the cycle order, and that `ZeroGrad` runs per batch, not per epoch |
| Learning rate never decays | Call `SetEpoch(epoch, num_epochs)` each epoch; `lr_min_frac = 1` also means constant |
| Second run behaves oddly | `Reset()` the Training — Adam moments and step count persist |
| Output saturates at ±1 | `tanh_last` was set to true somewhere — the default (false) writes the raw accumulator |
| `RestoreBest` did nothing | `restore_best` must be true in the config, and `Observe` must have seen a finite metric |
| `Observe` never snapshots | It wants **lower-is-better, strictly** — feed a loss or error rate, not an accuracy |
| Weights load rejected | `LoadWeights` requires the exact current weight count — same dim, z_max, and span |

## Model persistence

The library has no file format of its own: a model is fully described by
its configuration plus the weight vector.

| Mechanism | What is stored | Optimizer state? |
|-----------|----------------|------------------|
| `core->Weights()` / `core->LoadWeights(...)` | All weights, z-major, verbatim | **No** (Adam moments, step count, best snapshot live in `Training`) |

Write the vector with any format you like; on load, reconstruct the Core
from the **same config** (dim, z_max, gather_span decide the weight count
and layout) and `LoadWeights` the exact-length array. `Forward` on the
reloaded instance reproduces the original bit for bit. To continue
training, construct a fresh `Training` — Adam starts cold.

The Raman examples carry a ready-made recipe: a small header + weights
file format with a config check on load
([BaselineExtractor](../examples/RamanBaseline/BaselineExtractor.cpp)).
The Python surface pickles the same thing (config + weights).

## Limitations

- One `Core` + `Training` pair is **not thread-safe** for concurrent
  public calls — treat it as exclusive to one thread of control. Separate
  pairs on separate threads are independent; everything is single-threaded
  by design, so parallelism belongs to the host (the `AccumulateGrad`
  data-parallel pattern above is the worked form).
- The Core must **outlive** the Training (`Training` holds a reference),
  and neither class is copyable or movable — `Core::Create` hands you a
  `unique_ptr`; heap-allocate `Training` too if it must change hands.
- Per-sample forward cost is `z_max × N × dim × gather_span`
  multiply-adds plus a tanh per vertex per depth; `Backward` is roughly
  2× the forward.
- Memory: weights are `N × dim × gather_span × z_max` floats; the state
  stack is `(z_max + gather_span) × N`. `Training` adds four weight-sized
  buffers (gradient, two Adam moments, and — with `restore_best` — the
  snapshot) plus one state-sized buffer. The dim-12 Raman run is about
  1.6 M weights, so roughly 30 MB all in.
- There is no built-in noise, augmentation, or packing helper — hosts add
  those around the loop (`examples/common/` has optional recipes).

## Dependencies

| Layer | What |
|-------|------|
| Library | C++ standard library only |
| Build | C++23 compiler, CMake ≥ 3.20 |
| Examples | Same, plus the datasets described in [examples/README.md](../examples/README.md) |

There are no third-party dependencies.
