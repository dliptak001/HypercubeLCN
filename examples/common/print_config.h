// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

// Example-only config banners for the Core pipeline (ASCII for Windows
// consoles).

#include "Core.h"
#include "Training.h"

#include <cstdio>

namespace lcn_ex {

inline void PrintCoreConfig(const Core& core)
{
    std::printf(
        "core: dim=%zu N=%zu z_max=%zu gather_span=%zu tanh_last=%s "
        "seed=%llu |w|=%zu\n",
        core.Dim(), core.N(), core.ZMax(), core.GatherSpan(),
        core.TanhLast() ? "true" : "false",
        static_cast<unsigned long long>(core.Seed()),
        core.Weights().size());
    std::fflush(stdout);
}

inline void PrintTrainingConfig(const TrainingConfig& t, int epochs, int batch)
{
    std::printf(
        "train: epochs=%d batch=%d lr=%.6g lr_min_frac=%.6g lr_decay_epochs=%d "
        "restore_best=%s beta1=%.6g beta2=%.6g eps=%.6g\n",
        epochs, batch,
        static_cast<double>(t.lr),
        static_cast<double>(t.lr_min_frac),
        t.lr_decay_epochs,
        t.restore_best ? "true" : "false",
        static_cast<double>(t.beta1),
        static_cast<double>(t.beta2),
        static_cast<double>(t.eps));
    std::fflush(stdout);
}

} // namespace lcn_ex
