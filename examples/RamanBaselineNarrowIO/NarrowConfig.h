// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "BaselineExtractor.h"

// The narrow-IO experiment: same task, same 2048-wide IO, but the cube
// is dim 12 (N = 4096). The spectrum occupies addresses 0 .. 2047 (the
// bit-11 = 0 subcube); the other 2048 vertices are zero-fed on input
// and unconstrained by the loss — free hidden units. Depth, span, seed,
// schedule, epochs, and batch all match the dim-11 baseline, so the
// only variable is the extra capacity.
constexpr size_t kNarrowDim = 12;
constexpr size_t kNarrowN = size_t{1} << kNarrowDim;

inline CoreConfig MakeNarrowCoreConfig()
{
    CoreConfig cfg = MakeCoreConfig();
    cfg.dim = kNarrowDim;
    return cfg;
}

inline constexpr const char* kNarrowModelStem =
    "C:/HypercubeLCN/RamanModels/narrow12";
inline constexpr const char* kNarrowExtractDir =
    "C:/HypercubeLCN/RamanModels/extracted_narrow";
