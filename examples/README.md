# HypercubeLCN examples

Worked applications of the `Core` / `Training` library
([docs/forward.md](../docs/forward.md),
[docs/training.md](../docs/training.md)). Five build targets from
three examples — the dim-11 Raman baseline, its dim-12 narrow-IO
variant, and MNIST classification:

| Program | What it does |
|---------|--------------|
| `lcn_raman` | Train the baseline extractor, score it, save the weights |
| `lcn_raman_extract` | Load saved weights, write selected predictions to disk |
| `lcn_raman_narrow` | Same task on a dim-12 cube with 2048-wide IO |
| `lcn_raman_narrow_extract` | Extraction for the narrow-IO model |
| `lcn_mnist` | MNIST classification on a dim-11 cube, with a test-noise sweep |

Every knob is a `constexpr` at the top of a source file; there are no
command-line arguments. Change the constant, rebuild, run.

## The task: Raman baseline extraction

A Raman spectrum is a line of 2048 amplitudes: sharp molecular peaks
riding on a slow, unwanted background (fluorescence, mostly). The task
is regression from the spectrum to that background, so it can be
subtracted and the peaks measured cleanly.

The fit to the cube is direct. With `dim = 11` the hypercube has
`N = 2048` vertices, exactly one per spectral bin, so a spectrum is
already a field and no packing step exists. Neighboring addresses
differ in one bit, which for low bits means nearby wavenumbers; the
gather mixes information across the line while training shapes what
"background" means. The working configuration lives in
`MakeCoreConfig()` / `MakeTrainConfig()` in
[RamanBaseline/BaselineExtractor.h](RamanBaseline/BaselineExtractor.h):
`z_max = 8`, `gather_span = 4`, `tanh_last = false` (a baseline in
counts is not confined to (-1, 1), so the last depth stays linear),
cosine schedule from `5e-3` down to 5 percent of peak, `restore_best`
on.

The narrow-IO variant
([RamanBaselineNarrowIO](RamanBaselineNarrowIO/README.md)) runs the
same task on a dim-12 cube whose other 2048 vertices are free hidden
units — zero-fed on input, unconstrained by the loss — and beats the
baseline (1.688 vs 2.028 validation RMSE). Its README carries the
thesis and the results.

The MNIST example ([mnist](mnist/README.md)) is the first
classification task on the LCN: the 28 × 28 image resized onto two
32 × 32 planes (ink plus gradient) filling a dim-11 cube, the
10-class one-hot pattern striped across all 2048 output vertices
(vertex `v` targets class `v % 10`), prediction by argmax over the
per-class replica means, with per-epoch affine augmentation of the
raw digits. The default configuration scores 99.25 % clean test
accuracy; its README explains the striped-regression scheme and
records the reference run.

## Data

The dataset ships as a release asset of the
[HypercubeAIData](https://github.com/dliptak001/HypercubeAIData)
repository (about 75 MB zipped, roughly 1 GB unpacked). Download
[RamanSpectraLCOHard.zip](https://github.com/dliptak001/HypercubeAIData/releases/download/v1.0/RamanSpectraLCOHard.zip)
and unpack it into `C:\HypercubeAI\data`, so the tree lands at:

```text
C:\HypercubeAI\data\RamanSpectraLCOHard\
  Training\      0.data.txt, 0.label.txt, 1.data.txt, ...
  Validation\    same layout
```

All four programs read it in place from that path, via `kRamanDataRoot` in
[RamanBaseline/RamanPaths.h](RamanBaseline/RamanPaths.h).

Each `.data.txt` is one spectrum (2048 comma-separated floats, one
line); the matching `.label.txt` is its ground-truth baseline.

Normalization is per spectrum: the input's min/max maps it to
`[-1, 1]`, and the label is mapped with the *input's* min and range,
never its own, so the inverse transform of a prediction lands back in
real counts. Training loss is reported in normalized units
(`train_loss (norm)`), while the scoring RMSE is in counts
(`eval_rmse (counts)`) - the two columns of the epoch log.

## `lcn_raman`: train and score

Loads `kTrainSamples` from `Training` and `kTestSamples` from
`Validation` (constants at the top of
[RamanBaseline/lcn_raman.cpp](RamanBaseline/lcn_raman.cpp);
0 means the whole split), then trains for `kEpochs` epochs of
minibatch size `kBatch`. After every epoch it prints the training loss
and the RMSE over the eval pack; that RMSE is also the `restore_best`
metric, so the run ends on the best-eval weights, not the last-epoch
ones.

After training it saves the weights to `kRamanModelStem` (a `.w` file
under `C:/HypercubeLCN/RamanModels/`), reloads them into a fresh
extractor, and checks the two agree on a probe spectrum before
trusting the file. The final line reports RMSE over the full train and
validation splits.

Set `kSkipTrain = true` to skip training and score an already-saved
`.w` file instead.

## `lcn_raman_extract` and the plot

Training prints aggregate RMSE; to actually look at predictions, run
`lcn_raman_extract`. Its knobs sit at the top of
[RamanBaseline/lcn_raman_extract.cpp](RamanBaseline/lcn_raman_extract.cpp):
which split, which spectrum indices, and the output directory. It loads
the saved weights, predicts a baseline for each listed spectrum, and
writes one `{index}.pred.txt` per spectrum plus a `manifest.txt`
recording the stem, split, and indices.

[RamanBaseline/plot_extracted.py](RamanBaseline/plot_extracted.py)
reads that manifest - no second index list to keep in sync - and
overlays spectrum, ground-truth baseline, and prediction for each
extracted index into `extracted_baselines.png` (needs numpy and
matplotlib).

The loop, end to end: train once, pick indices to inspect, extract,
plot, adjust knobs, repeat.

## Folder layout

```text
examples/
  README.md               this file
  common/
    print_config.h            run banners (Core / Training)
    pack_field.h              pack a short pattern into a length-N field
    find_data_dir.h           MNIST IDX location check
    worker_pool.h             persistent threads for data-parallel training
    SpatialEmbed.*            image -> cube embedding (HypercubeCNN port)
    SpatialAug.*              affine/elastic/noise augmentation (HypercubeCNN port)
  RamanBaseline/
    README.md                 the dim-11 baseline: task, results, figure
    lcn_raman.cpp             train-and-score entry point
    lcn_raman_extract.cpp     extraction entry point
    BaselineExtractor.*       Core + Training wrapper; the config lives here
    RamanDataset.*            split loading
    RamanNorm.h               the [-1, 1] normalization
    RamanScore.h              RMSE in counts
    RamanExtract.*            prediction writing and the manifest
    RamanPaths.h              data root and model stem
    plot_extracted.py         overlay plot
    extracted_baselines.png   the README's figure
  RamanBaselineNarrowIO/
    README.md                 the dim-12 narrow-IO experiment and results
    NarrowConfig.h            dim-12 config, model stem, extract dir
    lcn_raman_narrow.cpp      train-and-score entry point
    lcn_raman_narrow_extract.cpp   extraction entry point
    plot_extracted.py         overlay plot (reads extracted_narrow)
    extracted_baselines_narrowIO.png   the README's figure
  mnist/
    README.md                 classification as striped regression
    lcn_mnist.cpp             train, score, and noise-sweep entry point
    mnist_idx.h               MNIST IDX loader
```

The narrow-IO example reuses the baseline folder's helpers
(`BaselineExtractor`, dataset, scoring, extraction) via the include
path; only its config and entry points are its own.
