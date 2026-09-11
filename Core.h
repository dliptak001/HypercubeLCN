// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/// @brief Knobs for @ref Core. All fixed at construction.
struct CoreConfig
{
    /// Hypercube dimension; N = 2^dim vertices. Must be in [4, 24].
    size_t dim = 8;
    /// Seed for the initial weight draw (normal, Xavier-style scale).
    uint64_t seed = 934791766227647176;
    /// History depth; 0 = use dim, else must be >= 2.
    /// 2 = self-dependence possible, dim = antipodal reach.
    size_t z_max = 0;
    /// Lookback window width in fields. Must be in [2, 6].
    size_t gather_span = 2;
    /// If true, the last depth applies tanh like every other depth,
    /// confining the output to (-1, 1). Default false: the last depth
    /// writes the raw accumulator — what squared-error training wants,
    /// even against ±1 targets (tanh only reaches ±1 asymptotically).
    bool tanh_last = false;
};

class Training;

/// @brief The network: a locally connected net on a Boolean hypercube.
///
/// Owns the weights and runs the forward pass: one length-N field in,
/// one length-N field out, with z_max intermediate fields written on the
/// same cube in between.
///
/// One instance is not thread-safe for concurrent public calls.
class Core
{
    friend class Training;

public:
    /// @brief Validate @p cfg, allocate state, and draw initial weights.
    /// @throws std::invalid_argument if @c dim is not in [4, 24],
    ///         resolved @c z_max is < 2 or absurdly large (weight count
    ///         would overflow), or @c gather_span is not in [2, 6].
    /// @throws std::bad_alloc (or std::length_error) if the state or weight
    ///         buffers do not fit in memory (large @c dim / @c z_max /
    ///         @c gather_span).
    static std::unique_ptr<Core> Create(const CoreConfig& cfg)
    {
        return std::unique_ptr<Core>(new Core(cfg));
    }

    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;
    Core(Core&&) = delete;
    Core& operator=(Core&&) = delete;

    /// @brief Run the forward pass on one length-N input field.
    /// The result is available from @ref Output until the next Forward.
    /// @throws std::invalid_argument if @p input_field is not length N.
    void Forward(std::span<const float> input_field);

    /// @brief Pointer form of @ref Forward. @p input_field must point at
    /// N floats; the length cannot be checked.
    /// @throws std::invalid_argument if @p input_field is null.
    void Forward(const float* input_field);

    /// Output field of the most recent @ref Forward. Length N; zeros
    /// before the first call. Valid until the next Forward.
    [[nodiscard]] const std::vector<float>& Output() const { return o_; }

    [[nodiscard]] size_t Dim() const { return dim_; }
    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] uint64_t Seed() const { return rng_seed_; }
    [[nodiscard]] size_t ZMax() const { return z_max_; }
    [[nodiscard]] size_t GatherSpan() const { return gather_span_; }
    [[nodiscard]] bool TanhLast() const { return tanh_last_; }

    /// The resolved configuration (z_max = 0 already replaced by dim).
    [[nodiscard]] CoreConfig Config() const
    {
        return CoreConfig{.dim = dim_, .seed = rng_seed_, .z_max = z_max_,
                          .gather_span = gather_span_, .tanh_last = tanh_last_};
    }

    /// All trainable weights, z-major: depth, vertex, axis, tap.
    /// Length N * dim * gather_span * z_max.
    [[nodiscard]] const std::vector<float>& Weights() const { return w_; }

    /// @brief Replace all weights (e.g. from a snapshot or a file).
    /// Invalidates any pending gradient state: the next Training::Backward
    /// needs a fresh Forward + Loss taken against the new weights.
    /// @throws std::invalid_argument if @p weights is not exactly
    ///         @ref Weights().size() long.
    void LoadWeights(std::span<const float> weights);

    /// @brief Pointer form of @ref LoadWeights.
    /// @throws std::invalid_argument if @p data is null or @p count
    ///         is not @ref Weights().size().
    void LoadWeights(const float* data, size_t count);

private:
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    explicit Core(const CoreConfig& cfg);

    uint64_t forward_serial_ = 0; // bumped by every Forward; Training matches
                                  // its Loss to it (stale-gradient guard)
    uint64_t rng_seed_ = 0;
    size_t dim_ = 0;
    size_t n_ = 0;
    size_t z_max_ = 0;        // weight depths; z runs 0 .. z_max_-1
    size_t gather_span_ = 0;
    bool tanh_last_ = false;
    std::vector<float> s_;
    std::vector<float> w_;    // z-major, uniform tables: dim axes x span taps per vertex
    std::vector<float> o_;    // output
};
