// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "Core.h"
#include "Training.h"

#include <memory>
#include <span>
#include <string>

struct RamanSplit;

// IO geometry: every spectrum, label, and prediction is kN wide. The
// cube itself may be larger (a wider CoreConfig); IO then occupies
// addresses 0 .. kN-1 and the rest are free hidden vertices.
constexpr size_t kDim = 11;
constexpr size_t kN = size_t{1} << kDim;
constexpr int kEpochs = 100;
constexpr int kBatch = 48;
constexpr uint64_t kShuffleSeed = 1;

// Data-parallel training workers (worker_pool.h): each owns a Core clone
// + Training and handles a strided share of every batch; the master
// merges gradients and takes the single Adam step. 0 = half the logical
// CPUs (~physical cores on SMT machines). Deterministic for a fixed
// value; float summation order differs from the old serial runs.
constexpr size_t kThreads = 0;


inline CoreConfig MakeCoreConfig()
{
    return CoreConfig{
        .dim = kDim,
        .seed = 934791766227647176,
        .z_max = 8,
        .gather_span = 4,
        .tanh_last = false,
    };
}

inline TrainingConfig MakeTrainConfig()
{
    return TrainingConfig{
        .lr = 5e-3f,
        .lr_min_frac = 0.05f, // 1 = constant.
        .lr_decay_epochs = 0, // 0 = kEpochs
        .restore_best = true,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
    };
}

class BaselineExtractor
{
public:
    // Return the restore-best metric (lower wins). Raman returns eval RMSE.
    using EpochTick = float (*)(int epoch, float train_loss);

    BaselineExtractor();
    BaselineExtractor(const CoreConfig& core_cfg, const TrainingConfig& train_cfg);
    ~BaselineExtractor() = default;

    BaselineExtractor(const BaselineExtractor&) = delete;
    BaselineExtractor& operator=(const BaselineExtractor&) = delete;
    BaselineExtractor(BaselineExtractor&&) = delete;
    BaselineExtractor& operator=(BaselineExtractor&&) = delete;

    [[nodiscard]] size_t Dim() const { return core_->Dim(); }
    [[nodiscard]] size_t N() const { return core_->N(); }
    [[nodiscard]] const CoreConfig& core_config() const { return core_cfg_; }
    [[nodiscard]] const TrainingConfig& train_config() const { return train_cfg_; }
    [[nodiscard]] Core& core() { return *core_; }
    [[nodiscard]] const Core& core() const { return *core_; }

    void Train(const RamanSplit& split, EpochTick tick = nullptr);
    void Predict(std::span<const float> spectrum, std::span<float> baseline);

    [[nodiscard]] bool HasBest() const { return train_->HasBest(); }
    [[nodiscard]] int BestEpoch() const { return train_->BestEpoch(); }
    [[nodiscard]] float BestMetric() const { return train_->BestMetric(); }

    void SaveWeights(const std::string& path_stem) const;
    void LoadWeights(const std::string& path_stem);

private:
    CoreConfig core_cfg_{};
    TrainingConfig train_cfg_{};
    std::unique_ptr<Core> core_;
    std::unique_ptr<Training> train_;
};
