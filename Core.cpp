// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>


static_assert(sizeof(size_t) >= 8,
              "HypercubeLCN requires a 64-bit build: the weight-count "
              "arithmetic (N * dim * span * z_max) overflows 32-bit size_t");

Core::Core(const CoreConfig& cfg)
    : rng_seed_(cfg.seed), dim_(cfg.dim),
      z_max_(cfg.z_max == 0 ? cfg.dim : cfg.z_max),
      gather_span_(cfg.gather_span), tanh_last_(cfg.tanh_last)
{
    if (dim_ < 4 || dim_ > 24)
        throw std::invalid_argument("Core::Core dim must be [4..24]");
    if (z_max_ < 2)
        throw std::invalid_argument("Core::Core z_max must be >= 2");
    if (gather_span_ < 2 || gather_span_ > 6)
        throw std::invalid_argument("Core::Core gather_span must be [2..6]");
    n_ = 1ULL << dim_;
    // n * dim * span <= 2^24 * 24 * 6 fits easily; z_max is the only factor
    // that could wrap the weight count and silently shrink the buffers
    if (z_max_ > std::numeric_limits<size_t>::max() / (n_ * dim_ * gather_span_))
        throw std::invalid_argument(
            "Core::Core z_max too large (weight count overflows size_t)");
    // prefix slots stay zero forever; Forward rewrites every other slot
    s_.assign(n_ * (z_max_ + gather_span_), 0.f);
    o_.assign(n_, 0.f);
    w_.resize(n_ * dim_ * gather_span_ * z_max_);
    std::mt19937_64 rng(rng_seed_);
    std::normal_distribution<float> dist(
        0.f, 1.f / std::sqrt(static_cast<float>(dim_ * gather_span_)));
    for (float& w : w_)
        w = dist(rng);
}

void Core::LoadWeights(std::span<const float> weights)
{
    if (weights.size() != w_.size())
        throw std::invalid_argument("Core::LoadWeights size mismatch");
    LoadWeights(weights.data(), weights.size());
}

void Core::LoadWeights(const float* data, size_t count)
{
    if (data == nullptr)
        throw std::invalid_argument("Core::LoadWeights data is null");
    if (count != w_.size())
        throw std::invalid_argument("Core::LoadWeights size mismatch");
    std::copy(data, data + count, w_.begin());
    // the state in s_ was produced by the old weights; force a fresh
    // Forward + Loss before any Backward (stale-gradient guard)
    ++forward_serial_;
}

void Core::Forward(std::span<const float> input_field)
{
    if (input_field.size() != n_)
        throw std::invalid_argument("Core::Forward input_field must be length N");
    Forward(input_field.data());
}

void Core::Forward(const float* input_field)
{
    if (input_field == nullptr)
        throw std::invalid_argument("Core::Forward input_field is null");

    ++forward_serial_;
    float* s = s_.data();
    const float* w = w_.data();
    const size_t n = n_;
    const size_t dim = dim_;
    const size_t span = gather_span_;
    std::copy(input_field, input_field + n, s + (span - 1) * n);

    // depth z reads slots z .. z+span-1, writes slot z+span.
    // v is innermost: consecutive vertices are contiguous.
    for (size_t z = 0; z < z_max_; ++z)
    {
        float* out = s + (z + span) * n;
        std::fill(out, out + n, 0.f);
        for (size_t axis = 0; axis < dim; ++axis)
        {
            const size_t mask = size_t{1} << axis;
            for (size_t k = 0; k < span; ++k)
            {
                const float* src = s + (z + k) * n;
                const float* wv = w + TapOffset(z, axis, k);
                for (size_t v = 0; v < n; ++v)
                    out[v] += wv[v] * src[v ^ mask];
            }
        }
        if (!((z + 1 == z_max_) && !tanh_last_))
        {
            for (size_t v = 0; v < n; ++v)
                out[v] = std::tanh(out[v]);
        }
    }

    std::copy(s + (z_max_ + span - 1) * n, s + (z_max_ + span) * n, o_.begin());
}
