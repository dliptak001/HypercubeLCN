// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstdint>
#include <span>
#include <vector>

class Core;

/// @brief Knobs for @ref Training. All fixed at construction.
struct TrainingConfig
{
    float lr = 5e-3f;           ///< Adam step size; cosine peak. Finite, > 0.
    float lr_min_frac = 1.f;    ///< Floor = lr * lr_min_frac. In [0, 1]; 1 = constant.
    int lr_decay_epochs = 0;    ///< Cosine horizon. 0 = SetEpoch's num_epochs. >= 0.
    bool restore_best = false;  ///< Snapshot weights on a new low score; restore at end.
    float beta1 = 0.9f;         ///< Adam first-moment decay. In [0, 1).
    float beta2 = 0.999f;       ///< Adam second-moment decay. In [0, 1).
    float eps = 1e-8f;          ///< Floor under the Adam denominator. Finite, > 0.
};

// Epoch 0 -> lr_max. Last epoch of the horizon -> lr_min.
// Same formula as Etalon / Cascade / WTF (HCNN cosine_lr).
[[nodiscard]] float CosineLR(float lr_max, float lr_min, int epoch, int num_epochs);

/// @brief Gradient training for a @ref Core: backprop through every depth,
/// stepped with Adam.
///
/// The per-batch cycle is: @ref ZeroGrad, then for each sample
/// Core::Forward → @ref Loss → @ref Backward, then @ref Adam. Loss reads
/// the output of the most recent Forward on the bound Core; Backward reads
/// the state that Forward and Loss left behind. @ref Backward accumulates
/// (sums) into the gradient, so the effective step scales with batch size.
///
/// Holds a reference to the Core passed at construction: the Core must
/// outlive the Training (bindings should pin the Core to the Training).
///
/// One instance is not thread-safe for concurrent public calls.
class Training
{
public:
    /// @throws std::invalid_argument if @p cfg violates the ranges
    ///         documented on @ref TrainingConfig.
    explicit Training(Core& core, const TrainingConfig& cfg);

    Training(const Training&) = delete;
    Training& operator=(const Training&) = delete;
    Training(Training&&) = delete;
    Training& operator=(Training&&) = delete;

    /// @brief Squared-error loss (0.5 * SSE) against the most recent
    /// Forward, and the gradient seed for @ref Backward.
    ///
    /// @p target may be shorter than N: only vertices
    /// 0 .. target.size()-1 carry targets and seed gradient; the rest of
    /// the output field is unconstrained (free hidden vertices).
    /// @throws std::invalid_argument if @p target is empty or longer than N.
    float Loss(std::span<const float> target);

    /// Pointer form: full-width loss, @p target must point at N floats.
    /// @throws std::invalid_argument if @p target is null.
    float Loss(const float* target);

    /// Pointer form of the masked loss; @p target points at
    /// @p target_count floats.
    /// @throws std::invalid_argument if @p target is null or
    ///         @p target_count is not in [1, N].
    float Loss(const float* target, size_t target_count);

    /// Backprop the most recent @ref Loss through every depth,
    /// accumulating into the gradient. Consumes the loss seed: another
    /// Backward first needs another @ref Loss (against the same Forward
    /// is fine — e.g. a second target on the same activations).
    /// @throws std::invalid_argument if no unconsumed Loss has been taken
    ///         against the most recent Core::Forward (stale-gradient
    ///         guard: Forward(a), Loss, Forward(b), Backward would
    ///         otherwise mix b's activations with a's loss seed; a second
    ///         Backward without a fresh Loss would re-read a dirtied seed).
    void Backward();

    /// Clear the accumulated gradient. Call at the start of each batch.
    void ZeroGrad();

    /// @brief Add an externally accumulated gradient into this instance's
    /// gradient, elementwise. The data-parallel merge: worker Trainings
    /// build private gradients (ZeroGrad / Loss / Backward on Core clones
    /// of the same shape), then one master accumulates each worker's
    /// @ref Grad and takes the single @ref Adam step.
    /// @throws std::invalid_argument if @p g is not @ref Grad().size() long.
    void AccumulateGrad(std::span<const float> g);

    /// One Adam step on the accumulated gradient. Mutates the weights, so
    /// any pending loss seed is invalidated: the next @ref Backward needs
    /// a fresh Core::Forward + @ref Loss (Core::LoadWeights and
    /// @ref RestoreBest invalidate the same way).
    void Adam();

    // Apply CosineLR for this epoch. Call once at the start of the epoch.
    // horizon = lr_decay_epochs if > 0, else num_epochs. horizon <= 0 keeps lr.
    void SetEpoch(int epoch, int num_epochs = 0);
    [[nodiscard]] float Lr() const { return lr_; }

    /// @brief Forget everything learned about the run: Adam moments and
    /// step count, the learning rate, and the best-weights snapshot.
    /// The Core's weights are left as they are.
    void Reset();

    // Lower metric wins (strict). No-op when restore_best is false.
    // RestoreBest writes the snapshot back; no-op if none.
    void Observe(float metric, int epoch);
    void RestoreBest();
    [[nodiscard]] bool HasBest() const { return !best_w_.empty(); }
    [[nodiscard]] int BestEpoch() const { return best_epoch_; }
    [[nodiscard]] float BestMetric() const { return best_metric_; }

    /// The accumulated gradient, same layout and length as Core::Weights().
    [[nodiscard]] const std::vector<float>& Grad() const { return dw_; }

    [[nodiscard]] const TrainingConfig& Config() const { return cfg_; }

private:
    Core& core_;
    TrainingConfig cfg_;
    float lr_;
    std::vector<float> dw_;
    std::vector<float> ds_;
    std::vector<float> m_;
    std::vector<float> v_;
    std::vector<float> best_w_;
    float best_metric_;
    int best_epoch_ = -1;
    int t_ = 0;
    // Core::forward_serial_ at the last Loss; kNoLoss = no unconsumed loss
    // seed (none taken yet, consumed by Backward, or invalidated by Adam)
    static constexpr uint64_t kNoLoss = ~0ULL;
    uint64_t loss_serial_ = kNoLoss;
};
