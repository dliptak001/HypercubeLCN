// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Training.h"
#include "Core.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

// Bit-level finiteness test. std::isfinite is unreliable under -ffast-math
// (-ffinite-math-only lets the compiler fold it to true); the exponent-bits
// check cannot be optimized away.
static bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

float CosineLR(float lr_max, float lr_min, int epoch, int num_epochs)
{
    if (num_epochs <= 1)
        return lr_max;
    if (epoch < 0)
        epoch = 0;
    if (epoch >= num_epochs)
        epoch = num_epochs - 1;

    const float progress =
        static_cast<float>(epoch) / static_cast<float>(num_epochs - 1);
    return lr_min + 0.5f * (lr_max - lr_min)
        * (1.f + std::cos(std::numbers::pi_v<float> * progress));
}

Training::Training(Core& core, const TrainingConfig& cfg)
    : core_(core), cfg_(cfg), lr_(cfg.lr),
      best_metric_(std::numeric_limits<float>::infinity())
{
    if (!FiniteBits(cfg.lr) || !(cfg.lr > 0.f))
        throw std::invalid_argument("Training::Training lr must be finite and > 0");
    if (!FiniteBits(cfg.lr_min_frac)
        || cfg.lr_min_frac < 0.f || cfg.lr_min_frac > 1.f)
        throw std::invalid_argument("Training::Training lr_min_frac must be [0..1]");
    if (cfg.lr_decay_epochs < 0)
        throw std::invalid_argument("Training::Training lr_decay_epochs must be >= 0");
    if (!FiniteBits(cfg.beta1) || !(cfg.beta1 >= 0.f) || !(cfg.beta1 < 1.f))
        throw std::invalid_argument("Training::Training beta1 must be [0..1)");
    if (!FiniteBits(cfg.beta2) || !(cfg.beta2 >= 0.f) || !(cfg.beta2 < 1.f))
        throw std::invalid_argument("Training::Training beta2 must be [0..1)");
    if (!FiniteBits(cfg.eps) || !(cfg.eps > 0.f))
        throw std::invalid_argument("Training::Training eps must be finite and > 0");

    dw_.assign(core_.w_.size(), 0.f);
    ds_.assign(core_.s_.size(), 0.f);
    m_.assign(core_.w_.size(), 0.f);
    v_.assign(core_.w_.size(), 0.f);
}

float Training::Loss(std::span<const float> target)
{
    if (target.empty() || target.size() > core_.n_)
        throw std::invalid_argument("Training::Loss target must be 1 .. N values");
    return Loss(target.data(), target.size());
}

float Training::Loss(const float* target)
{
    return Loss(target, core_.n_);
}

float Training::Loss(const float* target, size_t target_count)
{
    if (target == nullptr)
        throw std::invalid_argument("Training::Loss target is null");
    const size_t n = core_.n_;
    if (target_count == 0 || target_count > n)
        throw std::invalid_argument("Training::Loss target_count must be 1 .. N");

    loss_serial_ = core_.forward_serial_;
    std::ranges::fill(ds_, 0.f);
    const size_t base = (core_.z_max_ + core_.gather_span_ - 1) * n;
    float sum = 0.f;
    for (size_t v = 0; v < target_count; ++v)
    {
        const float d = core_.o_[v] - target[v];
        ds_[base + v] = d;
        sum += d * d;
    }
    return 0.5f * sum;
}

void Training::Backward()
{
    // Stale-gradient guard: the loss seed and the activations in s_ must
    // come from the same Forward, or the gradient is silent garbage.
    // kNoLoss also rejects a second Backward without a fresh Loss (the
    // pass below dirties ds_, so the seed is single-use).
    if (loss_serial_ != core_.forward_serial_)
        throw std::invalid_argument(
            "Training::Backward requires an unconsumed Loss against the "
            "most recent Core::Forward");

    const size_t n = core_.n_;
    const size_t dim = core_.dim_;
    const size_t span = core_.gather_span_;
    const size_t table_stride = dim * span;
    const float* s = core_.s_.data();
    const float* w = core_.w_.data();
    float* ds = ds_.data();
    float* dw = dw_.data();

    // reverse of Forward: depth z read slots z .. z+span-1, wrote slot z+span
    for (size_t z = core_.z_max_; z-- > 0;)
    {
        const float* w_z = w + z * n * table_stride;
        float* dw_z = dw + z * n * table_stride;
        const bool last = (z + 1 == core_.z_max_);

        // iterate over all vertices; every vertex has its own table
        for (size_t v = 0; v < n; ++v)
        {
            const float y = s[(z + span) * n + v];
            const float dact = (last && !core_.tanh_last_) ? 1.f : (1.f - y * y);
            const float incoming = ds[(z + span) * n + v] * dact;
            const float* w_v = w_z + v * table_stride;
            float* dw_v = dw_z + v * table_stride;

            // iterate over each nearest neighbor
            for (size_t axis = 0; axis < dim; ++axis)
            {
                const size_t v_nn = v ^ Core::NearestMask(axis);
                const float* w_axis = w_v + axis * span;
                float* dw_axis = dw_v + axis * span;

                // tap k read slot z+k; prefix taps see zeros, so their dw
                // stays zero and their ds spills into slots nothing reads
                for (size_t k = 0; k < span; ++k)
                {
                    dw_axis[k] += incoming * s[(z + k) * n + v_nn];
                    ds[(z + k) * n + v_nn] += incoming * w_axis[k];
                }
            }
        }
    }

    loss_serial_ = kNoLoss; // seed consumed; next Backward needs a new Loss
}

void Training::ZeroGrad()
{
    std::ranges::fill(dw_, 0.f);
}

void Training::AccumulateGrad(std::span<const float> g)
{
    if (g.size() != dw_.size())
        throw std::invalid_argument(
            "Training::AccumulateGrad gradient must be Grad().size() long");
    for (size_t i = 0; i < dw_.size(); ++i)
        dw_[i] += g[i];
}

void Training::Reset()
{
    std::ranges::fill(dw_, 0.f);
    std::ranges::fill(m_, 0.f);
    std::ranges::fill(v_, 0.f);
    t_ = 0;
    lr_ = cfg_.lr;
    best_w_.clear();
    best_metric_ = std::numeric_limits<float>::infinity();
    best_epoch_ = -1;
}

void Training::SetEpoch(int epoch, int num_epochs)
{
    const int horizon = (cfg_.lr_decay_epochs > 0) ? cfg_.lr_decay_epochs : num_epochs;
    if (horizon <= 0)
    {
        lr_ = cfg_.lr;
        return;
    }
    lr_ = CosineLR(cfg_.lr, cfg_.lr * cfg_.lr_min_frac, epoch, horizon);
}

void Training::Observe(float metric, int epoch)
{
    if (!cfg_.restore_best)
        return;
    if (!FiniteBits(metric))
        return;
    if (!(metric < best_metric_))
        return;
    best_w_ = core_.w_; // copy first: if it throws, metric/epoch stay honest
    best_metric_ = metric;
    best_epoch_ = epoch;
}

void Training::RestoreBest()
{
    if (!cfg_.restore_best || best_w_.empty())
        return;
    core_.LoadWeights(best_w_.data(), best_w_.size());
}

void Training::Adam()
{
    // stepping the weights makes the activations in s_ stale relative to
    // them; force a fresh Forward + Loss before the next Backward
    ++core_.forward_serial_;
    loss_serial_ = kNoLoss;
    ++t_;
    const float lr = lr_;
    const float b1 = cfg_.beta1;
    const float b2 = cfg_.beta2;
    const float eps = cfg_.eps;
    const float bc1 = 1.f - std::pow(b1, static_cast<float>(t_));
    const float bc2 = 1.f - std::pow(b2, static_cast<float>(t_));

    float* w = core_.w_.data();
    const size_t n = dw_.size();
    for (size_t i = 0; i < n; ++i)
    {
        const float g = dw_[i];
        m_[i] = b1 * m_[i] + (1.f - b1) * g;
        v_[i] = b2 * v_[i] + (1.f - b2) * g * g;
        const float mhat = m_[i] / bc1;
        const float vhat = v_[i] / bc2;
        w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
}
