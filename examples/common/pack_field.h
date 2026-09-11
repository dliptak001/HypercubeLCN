// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

// Example-only packing helpers (caller-owned field → length-N drive).
// Not part of the Core library. Raman does not pack: N = 2048 already.

#include <cstddef>
#include <span>
#include <stdexcept>

namespace lcn_ex {

/// Fill @p out (length N) with @p data in low addresses; pad the rest.
/// Requires data.size() <= out.size().
inline void PackPadLow(std::span<const float> data, std::span<float> out,
                       float pad_value = -1.0f)
{
    if (data.size() > out.size())
        throw std::invalid_argument(
            "PackPadLow: data longer than field N (need pad room or larger dim)");
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    for (size_t i = data.size(); i < out.size(); ++i)
        out[i] = pad_value;
}

} // namespace lcn_ex
