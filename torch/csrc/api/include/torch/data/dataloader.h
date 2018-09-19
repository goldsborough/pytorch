#pragma once

#include <torch/data/detail/data-shuttle.h>
#include <torch/data/detail/sequencers.h>
#include <torch/data/options.h>
#include <torch/data/samplers/random.h>

#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <cstddef>
#include <thread>
#include <type_traits>
#include <utility>

namespace torch {
namespace data {
template <
    typename DatasetType,
    typename SamplerType = samplers::RandomSampler<>>
class DataLoader {
 public:
  using Batch = typename DatasetType::BatchType;
  using IndexBatch = typename SamplerType::IndexBatchType;

  DataLoader(
      DatasetType dataset,
      DataLoaderOptions options,
      SamplerType sampler)
      : options_(options.coalesce()), sampler_(std::move(sampler)) {
    using namespace detail::sequencers;

    // clang-format off
    AT_CHECK(
        options_.batch_size_ <= dataset.size(),
        "Batch size (was ", options.batch_size_, ") ",
        "must not be larger than the dataset size (was ",
        dataset.size(), ")");
    // clang-format on

    if (options_.enforce_ordering_) {
      sequencer_ =
          torch::make_unique<OrderedSequencer<Result>>(*options_.max_jobs_);
    } else {
      sequencer_ = torch::make_unique<NoSequencer<Result>>();
    }

    if (options_.workers_ == 0) {
      dataset_ = std::move(dataset);
    }

    for (size_t w = 0; w < options_.workers_; ++w) {
      // Here we copy the dataset into the worker thread closure. Each worker
      // has its own copy of the dataset. This means the dataset must be
      // trivially copiable, or else we don't expect more than one worker to
      // be in use.
      workers_.emplace_back(
          [this, dataset] { this->worker_thread(std::move(dataset)); });
    }

    if (!options_.defer_prefetch_) {
      prefetch();
    }
  }

  ~DataLoader() {
    join();
  }

  void reset(bool prefetch = true) {
    shuttle_.drain();
    sampler_.reset();
    sequence_number_ = 0;
    if (prefetch) {
      this->prefetch();
    }
  }

  void prefetch(size_t requested_jobs) {
    while (requested_jobs-- > 0) {
      if (auto index_batch = get_index_batch()) {
        push_job(std::move(*index_batch));
      } else {
        shuttle_.exhausted();
      }
    }
  }

  void prefetch() {
    prefetch(*options_.max_jobs_);
  }

  at::optional<Batch> next() {
    at::optional<Batch> batch;
    if (options_.workers_ > 0) {
      // sequencer_->next(...).map(|r| r.batch)
      auto result = sequencer_->next(
          [this] { return shuttle_.pop_result(options_.timeout_); });
      if (result) {
        batch = std::move(result->batch);
      }
      prefetch(1);
    } else if (auto index_batch = get_index_batch()) {
      batch = dataset_->batch(std::move(*index_batch));
    }
    return batch;
  }

  void join() {
    if (joined_) {
      return;
    }
    shuttle_.drain();
    // Send one 'quit' message per worker. Since a worker dies (exits its
    // thread) after receiving this message, each `QuitWorker()` message will be
    // read by exactly one worker.
    for (size_t w = 0; w < options_.workers_; ++w) {
      push_job(QuitWorker());
    }
    for (auto& worker : workers_) {
      worker.join();
    }
    joined_ = true;
  }

  const DataLoaderOptions& options() const {
    return options_;
  }

 private:
  struct Sequenced {
    Sequenced() = default;
    Sequenced(size_t sqn) : sequence_number(sqn) {}
    size_t sequence_number;
  };

  struct QuitWorker {};
  // Job = Enum(QuitWorker, IndexBatch)
  struct Job : Sequenced {
    Job() = default;
    Job(QuitWorker q, size_t sqn) : Sequenced(sqn), quit(q) {}
    Job(IndexBatch&& i, size_t sqn)
        : Sequenced(sqn), index_batch(std::move(i)) {}
    at::optional<QuitWorker> quit;
    at::optional<IndexBatch> index_batch;
  };

  struct Result : Sequenced {
    Result() = default;
    Result(Batch&& b, size_t sqn) : Sequenced(sqn), batch(std::move(b)) {}
    Batch batch;
  };

  void worker_thread(DatasetType dataset) {
    while (true) {
      auto job = shuttle_.pop_job();
      if (job.quit) {
        break;
      }
      auto batch = dataset.batch(std::move(*job.index_batch));
      shuttle_.push_result({std::move(batch), job.sequence_number});
    }
  }

  at::optional<IndexBatch> get_index_batch() {
    auto indices = sampler_.next(options_.batch_size_);
    if (!indices ||
        (indices->size() < options_.batch_size_ && options_.drop_last_)) {
      return at::nullopt;
    }
    AT_ASSERT(!indices->empty());
    return indices;
  }

  template <typename T>
  void push_job(T&& value) {
    shuttle_.push_job({std::forward<T>(value), sequence_number_++});
  }

  const DataLoaderOptions options_;

  at::optional<DatasetType> dataset_;
  SamplerType sampler_;
  size_t sequence_number_{0};
  std::vector<std::thread> workers_;
  detail::DataShuttle<Job, Result> shuttle_;
  std::unique_ptr<detail::sequencers::Sequencer<Result>> sequencer_;
  bool joined_ = false;
};

template <typename DatasetType, typename SamplerType>
std::unique_ptr<DataLoader<DatasetType, SamplerType>> data_loader(
    DatasetType dataset,
    DataLoaderOptions options,
    SamplerType sampler) {
  return torch::make_unique<DataLoader<DatasetType, SamplerType>>(
      std::move(dataset), std::move(options), std::move(sampler));
}

template <
    typename DatasetType,
    typename SamplerType = samplers::RandomSampler<>,
    typename =
        torch::enable_if_t<std::is_constructible<SamplerType, size_t>::value>>
std::unique_ptr<DataLoader<DatasetType, SamplerType>> data_loader(
    DatasetType dataset,
    DataLoaderOptions options) {
  const auto size = dataset.size();
  return torch::make_unique<DataLoader<DatasetType, SamplerType>>(
      std::move(dataset), std::move(options), SamplerType(size));
}

} // namespace data
} // namespace torch
