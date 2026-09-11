// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

/// @file lcn_mnist.cpp
/// @brief MNIST → (augment) → spatial embed → LCN → held-out test.
///
/// The first classification task on the LCN, done as striped regression:
/// the image is embedded onto a dim-11 cube (kEmbedMode), and the 10-class
/// one-hot pattern is striped across the whole output field — vertex v
/// carries the target for class v % kClasses (+1 for the labeled class,
/// -1 otherwise), so all kN outputs train (full-width Training::Loss).
/// The predicted class is the argmax of the per-class means over each
/// class's ~kN/kClasses replica vertices.
///
/// Training is data-parallel (kThreads): each worker owns a Core clone +
/// Training and runs Forward/Loss/Backward on a strided share of every
/// batch; the master merges the worker gradients (AccumulateGrad) and
/// takes the single Adam step, then workers re-sync via LoadWeights.
/// Augmentation is seeded per (epoch, sample), so results do not depend
/// on the thread count — though they differ from the old serial stream.
///
/// The embedder and the optional per-epoch 2D augmentation are ports of
/// HypercubeCNN's HCNNSpatialEmbedder / HCNNSpatialAugmenter (see
/// examples/common/). Augmentation runs on the raw 28x28 each epoch,
/// then the augmented image is embedded — never the other way around.
/// The test set is embedded once, clean.
///
/// Reference run (50 epochs, clean test; details in README.md) —
/// DualPlaneResize + aug + zero-centered fields + striped readout,
/// i.e. the defaults as committed:
///   99.25 % test / 99.30 % train, best epoch 43, 30 min @ 12 threads
///
/// Optional test-only AWGN on the packed field (train clean). Demo knobs
/// kTestNoiseSweep / Start / End / Step. Off = clean test. On = train once,
/// then score start, start+step, ... while <= end and print a table.
///
/// Data: C:\HypercubeAI\data\mnist (uncompressed MNIST IDX files).

#include "Core.h"
#include "Training.h"
#include "SpatialAug.h"
#include "SpatialEmbed.h"
#include "find_data_dir.h"
#include "mnist_idx.h"
#include "print_config.h"
#include "worker_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Product knobs (edit here)
// =============================================================================

static constexpr size_t kDim = 11;
static constexpr size_t kN = size_t{1} << kDim;
static constexpr int kClasses = 10;
static constexpr int kEpochs = 50;
static constexpr int kBatch = 48;
static constexpr uint64_t kShuffleSeed = 1;

// Data-parallel workers: each owns a Core clone + Training and handles a
// strided share of every batch; one master merges gradients
// (Training::AccumulateGrad) and takes the single Adam step.
// 0 = half the logical CPUs (~physical cores on SMT machines).
static constexpr size_t kThreads = 0;

// DC-offset experiment: remap every packed field from the family
// convention [-1, 1] (background/pad = -1, ~87% of the field) to
// [0, 1] (background/pad = 0, ink = 1) just before it reaches the
// core, so the input is sparse instead of carrying a large negative
// DC offset. One transform at the embed boundary; targets stay +-1.
// Flip to false to back the experiment out exactly.
static constexpr bool kZeroCenterBackground = true;

// How the 28x28 lands on the cube (HCNNSpatialEmbedder port).
// DualPlaneResize at dim 11 is a perfect fit: 32x32 ink plane +
// 32x32 gradient plane = 2048 vertices, zero padding.
static constexpr auto kEmbedMode = hcnn::SpatialEmbedMode::DualPlaneResize;

// Per-epoch 2D augmentation on the raw image (HCNNSpatialAugmenter port).
// Defaults follow HypercubeCNN's MNIST example. false = train on clean.
static constexpr bool kAugEnabled = true;
static constexpr float kAugRotDegMax = 12.0f;
static constexpr float kAugScaleMin = 0.9f;
static constexpr float kAugScaleMax = 1.1f;
static constexpr int kAugShiftMax = 2;
static constexpr float kAugShearXMax = 0.15f;
static constexpr float kAugNoiseSigma = 0.03f;
static constexpr unsigned kAugSeed = 0xA46u;

static CoreConfig MakeCoreConfig()
{
    return CoreConfig{
        .dim = kDim,
        .seed = 234791766227647176,//934791766227647176,
        .z_max = 10,
        .gather_span = 4,
        .tanh_last = false,
    };
}

static TrainingConfig MakeTrainConfig()
{
    return TrainingConfig{
        .lr = 5e-3f,
        .lr_min_frac = 0.05f,
        .lr_decay_epochs = 0, // 0 = kEpochs
        .restore_best = true,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
    };
}

// =============================================================================
// Demo / task (not product config)
// =============================================================================

// MNIST train is 60000, test 10000. A short demo is 1000 / 500.
static constexpr int kTrainSamples = 60000;
static constexpr int kTestSamples = 10000;
static constexpr float kPad = -1.0f;
static constexpr int kImgSide = 28;
static constexpr int kImgPixels = kImgSide * kImgSide;
static constexpr double kMinTestAcc = 0.50;

static constexpr bool kTestNoiseSweep = false;
static constexpr float kTestNoiseStart = 0.0f;
static constexpr float kTestNoiseEnd = 1.0f;
static constexpr float kTestNoiseStep = 0.1f;
static constexpr unsigned kTestNoiseSeedBase = 0x7E57u;

// =============================================================================

static hcnn::SpatialEmbedder MakeEmbedder()
{
    hcnn::SpatialEmbedConfig cfg;
    cfg.dim = static_cast<int>(kDim);
    cfg.mode = kEmbedMode;
    cfg.pad_value = kPad;
    return hcnn::SpatialEmbedder(cfg);
}

static hcnn::SpatialAugmenter MakeAugmenter()
{
    if (!kAugEnabled)
        return hcnn::SpatialAugmenter(hcnn::SpatialAugConfig::None());
    hcnn::SpatialAugConfig cfg;
    cfg.rot_deg_max = kAugRotDegMax;
    cfg.scale_min = kAugScaleMin;
    cfg.scale_max = kAugScaleMax;
    cfg.shift_max = kAugShiftMax;
    cfg.shear_x_max = kAugShearXMax;
    cfg.noise_sigma = kAugNoiseSigma;
    cfg.border_value = kPad;
    return hcnn::SpatialAugmenter(cfg);
}

// =============================================================================
// Data-parallel workers
// =============================================================================

/// One worker: a private Core + Training (gradient accumulator and
/// scratch; its Adam moments are never stepped) plus per-thread buffers.
struct Worker
{
    std::unique_ptr<Core> core;
    std::unique_ptr<Training> train;
    std::vector<float> aug_img;
    std::vector<float> field;
    std::vector<float> y;
    float loss_sum = 0.f;
    size_t hits = 0;
};

/// The pool plus one Worker per thread, all built to the master's shape.
struct Workers
{
    lcn_ex::WorkerPool pool;
    std::vector<Worker> w;

    Workers(const CoreConfig& core_cfg, const TrainingConfig& train_cfg)
        : pool(kThreads)
    {
        w.reserve(pool.Size());
        for (size_t i = 0; i < pool.Size(); ++i)
        {
            Worker wk;
            wk.core = Core::Create(core_cfg);
            wk.train = std::make_unique<Training>(*wk.core, train_cfg);
            wk.aug_img.resize(static_cast<size_t>(kImgPixels));
            wk.field.resize(kN);
            wk.y.resize(kN);
            w.push_back(std::move(wk));
        }
    }
};

/// Per-sample augmentation seed (splitmix64 finalizer over kAugSeed,
/// epoch, sample index): the augmentation stream is a pure function of
/// the sample and epoch, so runs are independent of thread count.
static uint32_t AugSeedFor(int epoch, size_t sample_idx)
{
    uint64_t z = (static_cast<uint64_t>(kAugSeed) << 34)
        ^ (static_cast<uint64_t>(static_cast<uint32_t>(epoch)) << 20)
        ^ static_cast<uint64_t>(sample_idx);
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<uint32_t>(z ^ (z >> 31));
}

// =============================================================================

/// kZeroCenterBackground: [-1, 1] -> [0, 1] at the embed boundary.
static void RemapField(std::span<float> f)
{
    if constexpr (!kZeroCenterBackground)
        return;
    for (float& v : f)
        v = 0.5f * (v + 1.0f);
}

/// Clean (un-augmented) embed of a whole set, for scoring.
static void EmbedSet(const hcnn::SpatialEmbedder& emb,
                     const lcn_ex::MnistSet& ds,
                     std::vector<float>& fields,
                     std::vector<int>& labels)
{
    fields.assign(ds.size() * kN, 0.0f);
    labels.assign(ds.size(), 0);
    for (size_t i = 0; i < ds.size(); ++i)
    {
        labels[i] = ds.samples[i].label;
        emb.embed(ds.samples[i].pixels.data(), kImgSide, kImgSide,
                  fields.data() + i * kN);
        RemapField(std::span<float>(fields.data() + i * kN, kN));
    }
}

static int Classify(Core& core, std::span<const float> field)
{
    core.Forward(field.data());
    const auto& o = core.Output();
    // Mean output of each class's striped replicas (vertex v -> class
    // v % kClasses), argmax of the 10 means.
    double sum[kClasses] = {};
    size_t cnt[kClasses] = {};
    for (size_t v = 0; v < kN; ++v)
    {
        const int c = static_cast<int>(v % kClasses);
        sum[c] += o[v];
        ++cnt[c];
    }
    int best = 0;
    double best_mean = sum[0] / static_cast<double>(cnt[0]);
    for (int c = 1; c < kClasses; ++c)
    {
        const double mean = sum[c] / static_cast<double>(cnt[c]);
        if (mean > best_mean)
        {
            best_mean = mean;
            best = c;
        }
    }
    return best;
}

/// Parallel scoring: sync every worker to the master's weights, then
/// stride the samples across the pool.
static double Accuracy(Workers& ws, const Core& master,
                       std::span<const float> fields,
                       std::span<const int> labels)
{
    if (labels.empty() || fields.size() != labels.size() * kN)
        throw std::invalid_argument("Accuracy: field/label size mismatch");
    const size_t nw = ws.w.size();
    ws.pool.Run([&](size_t wi) {
        Worker& wk = ws.w[wi];
        wk.core->LoadWeights(master.Weights());
        size_t hits = 0;
        for (size_t i = wi; i < labels.size(); i += nw)
        {
            if (Classify(*wk.core, fields.subspan(i * kN, kN)) == labels[i])
                ++hits;
        }
        wk.hits = hits;
    });
    size_t hits = 0;
    for (const Worker& wk : ws.w)
        hits += wk.hits;
    return static_cast<double>(hits) / static_cast<double>(labels.size());
}

/// Striped one-hot in field convention: vertex v targets class
/// v % kClasses, +1 where that class is the label, -1 elsewhere.
/// Full kN width — every output vertex carries a target.
static void MakeTarget(int label, std::span<float> y)
{
    for (size_t v = 0; v < kN; ++v)
    {
        y[v] = (static_cast<int>(v % kClasses) == label)
            ? 1.0f : -1.0f;
    }
}

static void TrainClassifier(Core& core, Training& train, Workers& ws,
                            const lcn_ex::MnistSet& train_set,
                            const hcnn::SpatialAugmenter& aug,
                            const hcnn::SpatialEmbedder& emb,
                            std::span<const float> test_fields,
                            std::span<const int> test_labels)
{
    const size_t count = train_set.size();
    if (count == 0)
        throw std::invalid_argument("TrainClassifier: empty training set");

    const size_t batch = static_cast<size_t>(kBatch);
    // The merged gradient is a sum. Drop a short tail so every Adam step
    // sees `batch` samples. If the split is smaller than one batch, use
    // it as-is.
    size_t n_use = count - (count % batch);
    if (n_use == 0)
        n_use = count;

    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(kShuffleSeed);
    const size_t nw = ws.w.size();

    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        train.SetEpoch(epoch, kEpochs);
        std::shuffle(order.begin(), order.end(), rng);
        float loss_sum = 0.f;
        size_t nseen = 0;
        for (size_t start = 0; start < n_use; start += batch)
        {
            const size_t n = std::min(batch, n_use - start);
            // Workers: sync to the master weights, then each runs
            // Forward/Loss/Backward on its strided share of the batch,
            // accumulating a private gradient.
            ws.pool.Run([&](size_t wi) {
                Worker& wk = ws.w[wi];
                wk.core->LoadWeights(core.Weights());
                wk.train->ZeroGrad();
                wk.loss_sum = 0.f;
                for (size_t b = wi; b < n; b += nw)
                {
                    const size_t idx = order[start + b];
                    const auto& s = train_set.samples[idx];
                    // Fresh geometry + noise each epoch, then embed.
                    // Identity config (kAugEnabled false) makes apply()
                    // a pure copy.
                    std::mt19937 aug_rng(AugSeedFor(epoch, idx));
                    aug.apply(s.pixels.data(), wk.aug_img.data(),
                              kImgSide, kImgSide, aug_rng);
                    emb.embed(wk.aug_img.data(), kImgSide, kImgSide,
                              wk.field.data());
                    RemapField(wk.field);
                    wk.core->Forward(wk.field.data());
                    MakeTarget(s.label, wk.y);
                    wk.loss_sum += wk.train->Loss(wk.y.data()); // full kN
                    wk.train->Backward();
                }
            });
            // Master: merge in fixed worker order (deterministic for a
            // given thread count) and take the one Adam step.
            train.ZeroGrad();
            for (const Worker& wk : ws.w)
            {
                train.AccumulateGrad(wk.train->Grad());
                loss_sum += wk.loss_sum;
            }
            train.Adam();
            nseen += n;
        }
        const float mean_loss = loss_sum / static_cast<float>(nseen);
        const double acc = Accuracy(ws, core, test_fields, test_labels);
        std::printf("epoch=%d/%d train_loss=%.6f test_acc=%.4f\n",
                    epoch, kEpochs, mean_loss, acc);
        std::fflush(stdout);
        // restore_best wants lower-is-better; feed it the error rate.
        train.Observe(static_cast<float>(1.0 - acc), epoch);
    }
    train.RestoreBest();
}

// =============================================================================
// Test-only noise (ported from cascade_mnist)
// =============================================================================

static std::vector<float> MakeTestNoiseGrid()
{
    std::vector<float> grid;
    if (!kTestNoiseSweep)
        return grid;
    if (!(kTestNoiseStart >= 0.0f) || !std::isfinite(kTestNoiseStart)
        || !(kTestNoiseEnd >= 0.0f) || !std::isfinite(kTestNoiseEnd)
        || !std::isfinite(kTestNoiseStep) || !(kTestNoiseStep > 0.0f))
    {
        throw std::invalid_argument(
            "lcn_mnist: test noise sweep needs finite start/end >= 0 "
            "and step > 0");
    }
    if (kTestNoiseStart > kTestNoiseEnd)
    {
        throw std::invalid_argument(
            "lcn_mnist: kTestNoiseStart must be <= kTestNoiseEnd");
    }
    if (kTestNoiseEnd <= kTestNoiseStart)
    {
        grid.push_back(kTestNoiseStart);
        return grid;
    }
    const double n = std::floor(
        (static_cast<double>(kTestNoiseEnd) - kTestNoiseStart)
        / kTestNoiseStep + 1e-6) + 1.0;
    if (n > 1000.0)
    {
        throw std::invalid_argument(
            "lcn_mnist: test noise grid would exceed 1000 points");
    }
    const int count = static_cast<int>(n);
    grid.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        grid.push_back(kTestNoiseStart
            + kTestNoiseStep * static_cast<float>(i));
    }
    return grid;
}

/// In-place i.i.d. Gaussian on a packed field (no clamp). No-op if sigma <= 0.
static void AddTestFieldNoise(std::span<float> field, size_t sample_index,
                              float sigma)
{
    if (sigma <= 0.0f)
        return;
    std::mt19937 rng(kTestNoiseSeedBase
        + static_cast<unsigned>(sample_index) * 9973u);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (float& v : field)
        v += dist(rng);
}

static double ScoreNoisyTest(Workers& ws, const Core& core,
                             std::span<const float> clean_fields,
                             std::span<const int> labels, float sigma)
{
    if (sigma <= 0.0f)
        return Accuracy(ws, core, clean_fields, labels);

    std::vector<float> noisy(clean_fields.begin(), clean_fields.end());
    for (size_t i = 0; i < labels.size(); ++i)
    {
        AddTestFieldNoise(
            std::span<float>(noisy.data() + i * kN, kN), i, sigma);
    }
    return Accuracy(ws, core, noisy, labels);
}

// =============================================================================

int main()
{
    int exit_code = 1;
    try
    {
        // PadLow / PadLowCenter need room for the native image; the resize
        // modes always fit. Runtime plan() validates the rest.
        static_assert(kEmbedMode == hcnn::SpatialEmbedMode::ResizeToFit
                          || kEmbedMode == hcnn::SpatialEmbedMode::DualPlaneResize
                          || kN >= kImgPixels,
                      "pad embed needs N >= 784 (dim >= 10)");

        const auto data_dir = lcn_ex::FindMnistDataDir(nullptr);
        std::printf("lcn_mnist: loading IDX from %s\n",
                    data_dir.string().c_str());
        std::fflush(stdout);

        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "lcn_mnist: kTrainSamples / kTestSamples must be >= 0 "
                "(0 = whole file)");
        }
        const auto train_set = lcn_ex::LoadMnist(
            (data_dir / "train-images-idx3-ubyte").string(),
            (data_dir / "train-labels-idx1-ubyte").string(),
            static_cast<size_t>(kTrainSamples));
        const auto test_set = lcn_ex::LoadMnist(
            (data_dir / "t10k-images-idx3-ubyte").string(),
            (data_dir / "t10k-labels-idx1-ubyte").string(),
            static_cast<size_t>(kTestSamples));

        const auto emb = MakeEmbedder();
        const auto aug = MakeAugmenter();
        const auto plan = emb.plan(kImgSide, kImgSide);

        const char* mode_name =
            kEmbedMode == hcnn::SpatialEmbedMode::PadLow ? "PadLow"
            : kEmbedMode == hcnn::SpatialEmbedMode::PadLowCenter ? "PadLowCenter"
            : kEmbedMode == hcnn::SpatialEmbedMode::ResizeToFit ? "ResizeToFit"
            : "DualPlaneResize";
        std::printf("lcn_mnist: embed=%s train=%zu test=%zu pad=%g "
                    "plane_side=%d occupied=%d\n",
                    mode_name, train_set.size(), test_set.size(),
                    static_cast<double>(kPad),
                    plan.plane_side, plan.pattern_length);
        std::printf("lcn_mnist: io: %d classes striped over all %zu "
                    "vertices (vertex v -> class v %% %d, ~%zu replicas "
                    "each), no free hidden vertices\n",
                    kClasses, kN, kClasses,
                    kN / static_cast<size_t>(kClasses));
        std::printf("lcn_mnist: field=%s\n",
                    kZeroCenterBackground
                        ? "[0,1] background=0 (kZeroCenterBackground)"
                        : "[-1,1] background=-1 (family convention)");
        if (kAugEnabled)
        {
            std::printf("lcn_mnist: aug: rot=%g scale=[%g,%g] shift=%d "
                        "shear_x=%g noise=%g seed=0x%X\n",
                        static_cast<double>(kAugRotDegMax),
                        static_cast<double>(kAugScaleMin),
                        static_cast<double>(kAugScaleMax),
                        kAugShiftMax,
                        static_cast<double>(kAugShearXMax),
                        static_cast<double>(kAugNoiseSigma),
                        kAugSeed);
        }
        else
        {
            std::printf("lcn_mnist: aug: off\n");
        }

        const auto noise_grid = MakeTestNoiseGrid();
        if (kTestNoiseSweep)
        {
            std::printf(
                "lcn_mnist: test_noise=sweep start=%g end=%g step=%g "
                "(%zu points) seed_base=0x%X (train once, then each sigma)\n",
                static_cast<double>(kTestNoiseStart),
                static_cast<double>(kTestNoiseEnd),
                static_cast<double>(kTestNoiseStep),
                noise_grid.size(), kTestNoiseSeedBase);
        }
        else
        {
            std::printf("lcn_mnist: test_noise=off\n");
        }
        std::fflush(stdout);

        // Clean embeds, used for scoring only; training re-embeds each
        // sample from its augmented image every epoch.
        std::vector<float> train_fields;
        std::vector<int> train_labels;
        std::vector<float> test_fields;
        std::vector<int> test_labels;
        EmbedSet(emb, train_set, train_fields, train_labels);
        EmbedSet(emb, test_set, test_fields, test_labels);

        const CoreConfig core_cfg = MakeCoreConfig();
        const TrainingConfig train_cfg = MakeTrainConfig();
        auto core = Core::Create(core_cfg);
        Training train(*core, train_cfg);
        Workers workers(core_cfg, train_cfg);

        lcn_ex::PrintCoreConfig(*core);
        lcn_ex::PrintTrainingConfig(train_cfg, kEpochs, kBatch);
        std::printf("lcn_mnist: threads=%zu (kThreads=%zu)\n",
                    workers.pool.Size(), kThreads);
        std::fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        TrainClassifier(*core, train, workers, train_set, aug, emb,
                        test_fields, test_labels);
        const auto t1 = std::chrono::steady_clock::now();
        const double train_secs =
            std::chrono::duration<double>(t1 - t0).count();
        std::printf("trained time=%.1fs\n", train_secs);
        if (train_cfg.restore_best && train.HasBest())
        {
            std::printf("restore_best epoch=%d error=%.4f\n",
                        train.BestEpoch(),
                        static_cast<double>(train.BestMetric()));
        }
        std::fflush(stdout);

        const double train_acc =
            Accuracy(workers, *core, train_fields, train_labels);
        double test_acc = -1.0;
        if (kTestNoiseSweep)
        {
            std::printf("train_acc=%.4f; test noise sweep (%zu sigmas)\n",
                        train_acc, noise_grid.size());
            std::printf("  sigma      test_acc    test_s\n");
            std::printf("  ---------  --------    ------\n");
            for (float sigma : noise_grid)
            {
                const auto s0 = std::chrono::steady_clock::now();
                const double acc = ScoreNoisyTest(
                    workers, *core, test_fields, test_labels, sigma);
                const auto s1 = std::chrono::steady_clock::now();
                std::printf("  %9.4f  %8.4f    %6.1f\n",
                            static_cast<double>(sigma), acc,
                            std::chrono::duration<double>(s1 - s0).count());
                std::fflush(stdout);
                if (sigma <= 0.0f)
                    test_acc = acc;
            }
        }
        else
        {
            test_acc = Accuracy(workers, *core, test_fields, test_labels);
            std::printf("train_acc=%.4f test_acc=%.4f\n",
                        train_acc, test_acc);
        }
        std::fflush(stdout);

        if (test_acc >= 0.0 && test_acc < kMinTestAcc)
        {
            std::fprintf(stderr,
                         "lcn_mnist: test accuracy too low (need >= %.2f)\n",
                         kMinTestAcc);
        }
        else
        {
            exit_code = 0;
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "lcn_mnist: %s\n", e.what());
        std::fprintf(stderr,
                     "Place uncompressed MNIST IDX files in "
                     "C:\\HypercubeAI\\data\\mnist:\n"
                     "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                     "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n");
    }
    return exit_code;
}
