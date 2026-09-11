// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"
#include "worker_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

// Weight-file format tag ('LCN' + format version 1).
constexpr char kMagic[4] = {'L', 'C', 'N', '1'};

void CheckRoundTrip(std::span<const float> spec)
{
    const auto nrm = RamanNorm::FromSpectrum(spec);
    std::vector<float> mapped(kN);
    std::vector<float> back(kN);
    nrm.Apply(spec, mapped);
    nrm.Invert(mapped, back);
    for (size_t j = 0; j < kN; ++j)
    {
        if (std::fabs(back[j] - spec[j]) > 1e-2f)
        {
            throw std::runtime_error(
                "BaselineExtractor: spectrum denorm round-trip failed");
        }
    }
}

} // namespace

BaselineExtractor::BaselineExtractor()
    : BaselineExtractor(MakeCoreConfig(), MakeTrainConfig())
{
}

BaselineExtractor::BaselineExtractor(const CoreConfig& core_cfg,
                                     const TrainingConfig& train_cfg)
    : core_cfg_(core_cfg), train_cfg_(train_cfg),
      core_(Core::Create(core_cfg)),
      train_(std::make_unique<Training>(*core_, train_cfg))
{
    // IO always lives on addresses 0 .. kN-1. A wider cube (dim > 11)
    // leaves the remaining vertices as free hidden units: zero-fed on
    // input, unconstrained by the loss.
    if (core_->N() < kN)
        throw std::logic_error("BaselineExtractor: cube must have N >= 2048");
}

void BaselineExtractor::Train(const RamanSplit& split, EpochTick tick)
{
    if (split.spectra.size() != split.count * kN
        || split.baselines.size() != split.count * kN)
    {
        throw std::invalid_argument(
            "BaselineExtractor::Train: count does not match buffer size");
    }
    if (split.count == 0)
        throw std::invalid_argument("BaselineExtractor::Train: empty split");
    if (kBatch < 1)
        throw std::invalid_argument("BaselineExtractor::Train: kBatch must be >= 1");

    RamanNorm::Check();
    CheckRoundTrip(split.Spectrum(0));

    const size_t batch = static_cast<size_t>(kBatch);
    // dw_ is a sum. Drop a short tail so every Adam step sees `batch`
    // samples. If the split is smaller than one batch, use it as-is.
    size_t n_use = split.count - (split.count % batch);
    if (n_use == 0)
        n_use = split.count;

    std::vector<size_t> order(split.count);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(kShuffleSeed);

    // Data-parallel workers: a private Core + Training each (the worker
    // Adam moments are never stepped), plus per-thread IO buffers.
    struct Worker
    {
        std::unique_ptr<Core> core;
        std::unique_ptr<Training> train;
        std::vector<float> x;
        std::vector<float> y;
        float loss_sum = 0.f;
    };
    lcn_ex::WorkerPool pool(kThreads);
    std::vector<Worker> workers;
    workers.reserve(pool.Size());
    for (size_t i = 0; i < pool.Size(); ++i)
    {
        Worker wk;
        wk.core = Core::Create(core_cfg_);
        wk.train = std::make_unique<Training>(*wk.core, train_cfg_);
        wk.x.assign(core_->N(), 0.f); // tail past kN stays zero
        wk.y.assign(kN, 0.f);
        workers.push_back(std::move(wk));
    }
    const size_t nw = workers.size();

    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        train_->SetEpoch(epoch, kEpochs);
        std::shuffle(order.begin(), order.end(), rng);
        float loss_sum = 0.f;
        size_t nseen = 0;
        for (size_t start = 0; start < n_use; start += batch)
        {
            const size_t n = std::min(batch, n_use - start);
            // Workers: sync to the master weights, then Forward/Loss/
            // Backward over a strided share of the batch, each into its
            // private gradient.
            pool.Run([&](size_t wi) {
                Worker& wk = workers[wi];
                wk.core->LoadWeights(core_->Weights());
                wk.train->ZeroGrad();
                wk.loss_sum = 0.f;
                for (size_t b = wi; b < n; b += nw)
                {
                    const auto spec = split.Spectrum(order[start + b]);
                    const auto lab = split.Baseline(order[start + b]);
                    const auto nrm = RamanNorm::FromSpectrum(spec);
                    nrm.Apply(spec, std::span(wk.x).first(kN));
                    nrm.Apply(lab, wk.y);
                    wk.core->Forward(wk.x.data());
                    wk.loss_sum += wk.train->Loss(wk.y.data(), kN);
                    wk.train->Backward();
                }
            });
            // Master: merge in fixed worker order (deterministic for a
            // given kThreads) and take the one Adam step.
            train_->ZeroGrad();
            float batch_loss = 0.f;
            for (const Worker& wk : workers)
            {
                train_->AccumulateGrad(wk.train->Grad());
                batch_loss += wk.loss_sum;
            }
            train_->Adam();
            loss_sum += batch_loss;
            nseen += n;
        }
        const float mean_loss = loss_sum / static_cast<float>(nseen);
        float metric = mean_loss;
        if (tick)
            metric = tick(epoch, mean_loss);
        train_->Observe(metric, epoch);
    }
    train_->RestoreBest();
}

void BaselineExtractor::Predict(std::span<const float> spectrum,
                                std::span<float> baseline)
{
    if (spectrum.size() != kN || baseline.size() != kN)
    {
        throw std::invalid_argument(
            "BaselineExtractor::Predict: spectrum and baseline size must equal kN");
    }

    const auto nrm = RamanNorm::FromSpectrum(spectrum);
    std::vector<float> xn(core_->N(), 0.f);
    nrm.Apply(spectrum, std::span(xn).first(kN));
    core_->Forward(xn.data());
    nrm.Invert(std::span<const float>(core_->Output()).first(kN), baseline);
}

void BaselineExtractor::SaveWeights(const std::string& path_stem) const
{
    if (path_stem.empty())
    {
        throw std::invalid_argument(
            "BaselineExtractor::SaveWeights: empty path_stem");
    }

    namespace fs = std::filesystem;
    const fs::path stem(path_stem);
    if (stem.has_parent_path())
        fs::create_directories(stem.parent_path());

    const fs::path dst = stem.string() + ".w";
    const fs::path tmp = stem.string() + ".w.writing";

    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out)
            throw std::runtime_error("cannot open " + tmp.string());
        const auto& w = core_->Weights();
        const uint32_t dim = static_cast<uint32_t>(core_->Dim());
        const uint32_t z_max = static_cast<uint32_t>(core_->ZMax());
        const uint32_t span = static_cast<uint32_t>(core_->GatherSpan());
        const uint64_t nw = static_cast<uint64_t>(w.size());
        out.write(kMagic, 4);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(&z_max), sizeof(z_max));
        out.write(reinterpret_cast<const char*>(&span), sizeof(span));
        out.write(reinterpret_cast<const char*>(&nw), sizeof(nw));
        out.write(reinterpret_cast<const char*>(w.data()),
                  static_cast<std::streamsize>(nw * sizeof(float)));
        if (!out)
            throw std::runtime_error("write failed " + tmp.string());
    }

    std::error_code ec;
    fs::remove(dst, ec);
    fs::rename(tmp, dst);
}

void BaselineExtractor::LoadWeights(const std::string& path_stem)
{
    if (path_stem.empty())
    {
        throw std::invalid_argument(
            "BaselineExtractor::LoadWeights: empty path_stem");
    }

    const std::filesystem::path src(path_stem + ".w");
    std::ifstream in(src, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + src.string());

    char magic[4] = {};
    uint32_t dim = 0;
    uint32_t z_max = 0;
    uint32_t span = 0;
    uint64_t nw = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&z_max), sizeof(z_max));
    in.read(reinterpret_cast<char*>(&span), sizeof(span));
    in.read(reinterpret_cast<char*>(&nw), sizeof(nw));
    if (!in || std::string(magic, magic + 4) != std::string(kMagic, kMagic + 4))
        throw std::runtime_error("LoadWeights: bad magic " + src.string());
    if (dim != core_->Dim())
        throw std::runtime_error("LoadWeights: dim mismatch");
    if (z_max != core_->ZMax())
        throw std::runtime_error("LoadWeights: z_max mismatch");
    if (span != core_->GatherSpan())
        throw std::runtime_error("LoadWeights: gather_span mismatch");
    if (nw != core_->Weights().size())
        throw std::runtime_error("LoadWeights: weight count mismatch");

    std::vector<float> w(static_cast<size_t>(nw));
    in.read(reinterpret_cast<char*>(w.data()),
            static_cast<std::streamsize>(nw * sizeof(float)));
    if (!in)
        throw std::runtime_error("LoadWeights: short read " + src.string());
    core_->LoadWeights(w.data(), w.size());
}
