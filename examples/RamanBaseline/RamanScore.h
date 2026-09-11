// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "BaselineExtractor.h"
#include "RamanDataset.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

/// Per-spectrum MSE of denormalized pred vs raw label.
/// Split RMSE = sqrt(mean of these).
inline double RamanPatternMse(std::span<const float> pred,
                              std::span<const float> label)
{
    if (pred.size() != label.size() || pred.empty())
    {
        throw std::invalid_argument(
            "RamanPatternMse: pred and label must be the same non-empty size");
    }
    double acc = 0.0;
    for (size_t j = 0; j < pred.size(); ++j)
    {
        const double e = static_cast<double>(label[j])
                         - static_cast<double>(pred[j]);
        acc += e * e;
    }
    return acc / static_cast<double>(pred.size());
}

inline double RamanRmse(BaselineExtractor& ex, const RamanSplit& split)
{
    if (split.count == 0)
        throw std::invalid_argument("RamanRmse: empty split");

    std::vector<float> pred(kN);
    double sum_mse = 0.0;
    for (size_t i = 0; i < split.count; ++i)
    {
        ex.Predict(split.Spectrum(i), pred);
        sum_mse += RamanPatternMse(pred, split.Baseline(i));
    }
    return std::sqrt(sum_mse / static_cast<double>(split.count));
}
