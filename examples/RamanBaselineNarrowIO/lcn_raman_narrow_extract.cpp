// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "NarrowConfig.h"
#include "RamanDataset.h"
#include "RamanExtract.h"
#include "RamanPaths.h"

#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr const char* kSplit = "Validation";
static constexpr int kIndices[] = {351, 352, 353, 354};

int main()
{
    int exit_code = 1;
    try
    {
        const std::span<const int> indices(kIndices);
        const auto split_dir =
            std::filesystem::path(kRamanDataRoot) / kSplit;
        const std::string stem(kNarrowModelStem);
        const std::filesystem::path out_dir(kNarrowExtractDir);

        BaselineExtractor ex(MakeNarrowCoreConfig(), MakeTrainConfig());
        LoadExtractor(ex, stem);
        const auto split = LoadRamanIndices(split_dir, indices);

        std::vector<float> preds(split.count * kN);
        ExtractSplit(ex, split, preds);
        WritePredictions(out_dir, indices, preds, stem, split_dir);

        std::printf("lcn_raman_narrow_extract: extracted n=%zu stem=%s -> %s\n",
                    split.count, stem.c_str(), out_dir.string().c_str());
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "lcn_raman_narrow_extract: %s\n", e.what());
    }
    return exit_code;
}
