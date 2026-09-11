// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"
#include "RamanPaths.h"
#include "RamanScore.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int kTrainSamples = 10000;
static constexpr int kTestSamples = 2000;
static constexpr int kEval = kTestSamples;
static constexpr bool kSkipTrain = false;

static BaselineExtractor* s_ex = nullptr;
static const RamanSplit* s_eval = nullptr;

static float EpochTick(int epoch, float train_loss)
{
    std::printf("epoch=%d/%d train_loss=%.6f (norm)", epoch, kEpochs, train_loss);
    float metric = train_loss;
    if (s_ex != nullptr && s_eval != nullptr)
    {
        const double rmse = RamanRmse(*s_ex, *s_eval);
        std::printf(" eval_rmse=%.6f (counts)", rmse);
        metric = static_cast<float>(rmse);
    }
    std::printf("\n");
    std::fflush(stdout);
    return metric;
}

int main()
{
    int exit_code = 1;
    try
    {
        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "kTrainSamples / kTestSamples must be >= 0 (0 = whole split)");
        }

        const std::filesystem::path root(kRamanDataRoot);
        const auto train = LoadRamanSplit(
            root / "Training", static_cast<size_t>(kTrainSamples));
        const auto test = LoadRamanSplit(
            root / "Validation", static_cast<size_t>(kTestSamples));

        if (train.count == 0)
            throw std::runtime_error("empty training split");

        RamanSplit eval;
        if (kEval > 0)
        {
            if (test.count < static_cast<size_t>(kEval))
            {
                throw std::invalid_argument(
                    "kEval larger than validation split");
            }
            eval.count = static_cast<size_t>(kEval);
            eval.spectra.assign(test.spectra.begin(),
                                test.spectra.begin() + eval.count * kN);
            eval.baselines.assign(test.baselines.begin(),
                                  test.baselines.begin() + eval.count * kN);
        }

        const std::string stem(kRamanModelStem);
        const CoreConfig core_cfg = MakeCoreConfig();
        const TrainingConfig train_cfg = MakeTrainConfig();

        BaselineExtractor ex(core_cfg, train_cfg);
        if (ex.N() != kN)
            throw std::logic_error("extractor N must equal kN");
        s_ex = &ex;
        s_eval = (kEval > 0) ? &eval : nullptr;

        std::printf("train=%zu val=%zu eval=%d skip_train=%s\n",
                    train.count, test.count, kEval,
                    kSkipTrain ? "true" : "false");
        lcn_ex::PrintCoreConfig(ex.core());
        lcn_ex::PrintTrainingConfig(train_cfg, kEpochs, kBatch);
        std::fflush(stdout);

        if (kSkipTrain)
        {
            ex.LoadWeights(stem);
            std::printf("loaded %s.w\n", stem.c_str());
            std::fflush(stdout);
        }
        else
        {
            const auto t0 = std::chrono::steady_clock::now();
            ex.Train(train, EpochTick);
            const auto t1 = std::chrono::steady_clock::now();
            const double train_secs =
                std::chrono::duration<double>(t1 - t0).count();
            const int train_h = static_cast<int>(train_secs / 3600.0);
            const int train_m =
                static_cast<int>((train_secs - 3600.0 * train_h) / 60.0);
            const double train_s =
                train_secs - 3600.0 * train_h - 60.0 * train_m;
            std::printf("trained time=%dh %dm %.2fs\n",
                        train_h, train_m, train_s);
            if (train_cfg.restore_best && ex.HasBest())
            {
                std::printf("restore_best epoch=%d metric=%.6f\n",
                            ex.BestEpoch(), static_cast<double>(ex.BestMetric()));
            }
            std::fflush(stdout);

            ex.SaveWeights(stem);
            {
                BaselineExtractor check(core_cfg, train_cfg);
                check.LoadWeights(stem);
                std::vector<float> a(kN), b(kN);
                const auto probe = train.Spectrum(0);
                ex.Predict(probe, a);
                check.Predict(probe, b);
                for (size_t i = 0; i < kN; ++i)
                {
                    if (std::fabs(a[i] - b[i]) > 1e-3f)
                    {
                        throw std::runtime_error(
                            "saved weights failed reload check");
                    }
                }
            }
            std::printf("saved %s.w\n", stem.c_str());
            std::fflush(stdout);
        }

        const double train_rmse = RamanRmse(ex, train);
        const double val_rmse = RamanRmse(ex, test);
        std::printf("train_rmse=%.6f val_rmse=%.6f\n",
                    train_rmse, val_rmse);
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
    }
    return exit_code;
}
