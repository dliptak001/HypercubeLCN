// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

// Persistent worker pool for the examples: T long-lived threads woken per
// Run() call, so a training loop can dispatch tens of thousands of small
// parallel regions (one per batch) without spawning threads each time.
// Example-only helper — the SDK surface (Core / Training) stays single-
// threaded; data-parallelism is composed outside it.

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace lcn_ex {

/// @brief T persistent threads; Run(fn) calls fn(worker_index) on every
/// worker concurrently and blocks until all return.
///
/// Run() is not reentrant and must be called from one thread at a time.
/// If any fn throws, the first exception is rethrown from Run() after all
/// workers finish the round.
class WorkerPool
{
public:
    /// @p threads 0 = half the logical CPUs (~physical cores on SMT
    /// machines), floored at 1.
    explicit WorkerPool(size_t threads = 0)
    {
        if (threads == 0)
        {
            threads = std::thread::hardware_concurrency() / 2;
            if (threads == 0)
                threads = 1;
        }
        workers_.reserve(threads);
        for (size_t i = 0; i < threads; ++i)
            workers_.emplace_back([this, i] { WorkerLoop(i); });
    }

    ~WorkerPool()
    {
        {
            std::lock_guard lock(mu_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& t : workers_)
            t.join();
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    [[nodiscard]] size_t Size() const { return workers_.size(); }

    /// Run @p fn(worker_index) on every worker; blocks until all return.
    void Run(const std::function<void(size_t)>& fn)
    {
        std::unique_lock lock(mu_);
        fn_ = &fn;
        pending_ = workers_.size();
        first_error_ = nullptr;
        ++generation_;
        cv_work_.notify_all();
        cv_done_.wait(lock, [this] { return pending_ == 0; });
        fn_ = nullptr;
        if (first_error_)
            std::rethrow_exception(first_error_);
    }

private:
    void WorkerLoop(size_t index)
    {
        uint64_t seen = 0;
        for (;;)
        {
            const std::function<void(size_t)>* fn = nullptr;
            {
                std::unique_lock lock(mu_);
                cv_work_.wait(lock, [this, seen] {
                    return stop_ || generation_ != seen;
                });
                if (stop_)
                    return;
                seen = generation_;
                fn = fn_;
            }
            std::exception_ptr err;
            try
            {
                (*fn)(index);
            }
            catch (...)
            {
                err = std::current_exception();
            }
            {
                std::lock_guard lock(mu_);
                if (err && !first_error_)
                    first_error_ = err;
                if (--pending_ == 0)
                    cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    const std::function<void(size_t)>* fn_ = nullptr;
    std::exception_ptr first_error_;
    uint64_t generation_ = 0;
    size_t pending_ = 0;
    bool stop_ = false;
};

} // namespace lcn_ex
