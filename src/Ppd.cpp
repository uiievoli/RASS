#include "Ppd.h"

#include <Eigen/Dense>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "Index.h"
#include "heap.hpp"
#include "utils.h"

namespace tribase {
namespace {

constexpr char PPD_MAGIC[8] = {'T', 'R', 'I', 'P', 'P', 'D', '1', '\0'};
constexpr uint32_t PPD_VERSION = 1;

template <typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void read_value(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
}

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace

void PpdIndex::build(const Index& index, size_t max_training_samples, uint64_t seed_value) {
    if (index.metric != MetricType::METRIC_L2 || index.d == 0 || index.nlist == 0 ||
        !index.lists) {
        throw std::invalid_argument("PPD requires a populated L2 IVF index");
    }
    size_t total = 0;
    for (size_t list_id = 0; list_id < index.nlist; ++list_id) {
        total += index.lists[list_id].list_size;
    }
    if (total < 2 || max_training_samples < 2) {
        throw std::invalid_argument("PPD PCA requires at least two training vectors");
    }
    d = index.d;
    nlist = index.nlist;
    seed = seed_value;
    training_samples = std::min(total, max_training_samples);

    // Deterministic stratified sample over the concatenated IVF lists.  The
    // paper does not publish a sampling rule; using all points remains possible
    // by setting max_training_samples >= database size.
    RowMatrix sample(static_cast<Eigen::Index>(training_samples),
                     static_cast<Eigen::Index>(d));
    // Pick one reproducible point from each equal-width stratum. This bounds
    // the sampling skew while still making --seed meaningful.
    std::vector<size_t> targets(training_samples);
    uint64_t state = seed_value;
    for (size_t i = 0; i < training_samples; ++i) {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        z ^= z >> 31U;
        const long double fraction =
            static_cast<long double>(z >> 11U) / static_cast<long double>(1ULL << 53U);
        targets[i] = std::min(
            total - 1,
            static_cast<size_t>((static_cast<long double>(i) + fraction) * total /
                                training_samples));
    }
    size_t flat_begin = 0;
    size_t output = 0;
    for (size_t list_id = 0; list_id < nlist && output < training_samples; ++list_id) {
        const IVF& list = index.lists[list_id];
        const size_t flat_end = flat_begin + list.list_size;
        while (output < training_samples) {
            const size_t target = targets[output];
            if (target >= flat_end) break;
            if (target >= flat_begin) {
                const size_t local = target - flat_begin;
                std::memcpy(sample.data() + output * d,
                            list.candidate_codes.get() + local * d,
                            d * sizeof(float));
                ++output;
            }
        }
        flat_begin = flat_end;
    }
    if (output != training_samples) {
        throw std::runtime_error("PPD stratified sampler produced an incomplete sample");
    }

    const Eigen::RowVectorXf mean_row = sample.colwise().mean();
    sample.rowwise() -= mean_row;
    Eigen::MatrixXf covariance =
        (sample.transpose() * sample) / static_cast<float>(training_samples);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("PPD PCA eigendecomposition failed");
    }
    mean.assign(mean_row.data(), mean_row.data() + d);
    basis.resize(d * d);
    for (size_t component = 0; component < d; ++component) {
        const Eigen::Index source = static_cast<Eigen::Index>(d - 1 - component);
        Eigen::VectorXf vector = solver.eigenvectors().col(source);
        for (Eigen::Index axis = 0; axis < vector.size(); ++axis) {
            if (std::abs(vector[axis]) > 1e-12f) {
                if (vector[axis] < 0.0f) vector = -vector;
                break;
            }
        }
        for (size_t axis = 0; axis < d; ++axis) {
            basis[component * d + axis] = vector[static_cast<Eigen::Index>(axis)];
        }
    }

    const Eigen::Map<const RowMatrix> basis_matrix(
        basis.data(), static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(d));
    const Eigen::Map<const Eigen::RowVectorXf> mean_matrix(
        mean.data(), static_cast<Eigen::Index>(d));
    const Eigen::RowVectorXf projected_mean = mean_matrix * basis_matrix.transpose();
    transformed_lists.clear();
    transformed_lists.resize(nlist);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        const IVF& list = index.lists[list_id];
        auto& destination = transformed_lists[list_id];
        destination.resize(list.list_size * d);
        if (list.list_size == 0) continue;
        const Eigen::Map<const RowMatrix> source(
            list.candidate_codes.get(), static_cast<Eigen::Index>(list.list_size),
            static_cast<Eigen::Index>(d));
        Eigen::Map<RowMatrix> transformed(
            destination.data(), static_cast<Eigen::Index>(list.list_size),
            static_cast<Eigen::Index>(d));
        transformed.noalias() = source * basis_matrix.transpose();
        transformed.rowwise() -= projected_mean;
    }
}

void PpdIndex::save(const std::string& path) const {
    prepareDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create PPD sidecar: " + path);
    out.write(PPD_MAGIC, sizeof(PPD_MAGIC));
    write_value(out, PPD_VERSION);
    write_value(out, d);
    write_value(out, nlist);
    write_value(out, training_samples);
    write_value(out, seed);
    out.write(reinterpret_cast<const char*>(mean.data()), mean.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(basis.data()), basis.size() * sizeof(float));
    for (const auto& list : transformed_lists) {
        const size_t values = list.size();
        write_value(out, values);
        out.write(reinterpret_cast<const char*>(list.data()), values * sizeof(float));
    }
    if (!out) throw std::runtime_error("Failed while writing PPD sidecar: " + path);
}

void PpdIndex::load(const std::string& path, const Index& index) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open PPD sidecar: " + path);
    char magic[sizeof(PPD_MAGIC)]{};
    uint32_t version = 0;
    in.read(magic, sizeof(magic));
    read_value(in, version);
    read_value(in, d);
    read_value(in, nlist);
    read_value(in, training_samples);
    read_value(in, seed);
    if (!in || std::memcmp(magic, PPD_MAGIC, sizeof(magic)) != 0 ||
        version != PPD_VERSION || d != index.d || nlist != index.nlist) {
        throw std::runtime_error("PPD sidecar is incompatible with the IVF index");
    }
    mean.resize(d);
    basis.resize(d * d);
    in.read(reinterpret_cast<char*>(mean.data()), mean.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(basis.data()), basis.size() * sizeof(float));
    transformed_lists.clear();
    transformed_lists.resize(nlist);
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        size_t values = 0;
        read_value(in, values);
        const size_t expected = index.lists[list_id].list_size * d;
        if (!in || values != expected) {
            throw std::runtime_error(
                std::format("PPD list {} has {} values; expected {}", list_id, values, expected));
        }
        transformed_lists[list_id].resize(values);
        in.read(reinterpret_cast<char*>(transformed_lists[list_id].data()),
                values * sizeof(float));
    }
    if (!in) throw std::runtime_error("Truncated PPD sidecar: " + path);
}

void PpdIndex::transform_query(const float* query, float* transformed) const {
    for (size_t component = 0; component < d; ++component) {
        float value = 0.0f;
#pragma omp simd reduction(+ : value)
        for (size_t axis = 0; axis < d; ++axis) {
            value += (query[axis] - mean[axis]) * basis[component * d + axis];
        }
        transformed[component] = value;
    }
}

PpdSearchStats searchPpd(const Index& index,
                         const PpdIndex& ppd,
                         size_t query_count,
                         const float* queries,
                         size_t k,
                         size_t nprobe,
                         size_t block_size,
                         bool triangle,
                         float* distances,
                         idx_t* labels) {
    if (index.metric != MetricType::METRIC_L2 || index.d != ppd.d ||
        index.nlist != ppd.nlist || block_size == 0 || k == 0) {
        throw std::invalid_argument("Invalid PPD search configuration");
    }
    nprobe = std::min(nprobe, index.nlist);
    if (triangle) {
        for (size_t list_id = 0; list_id < index.nlist; ++list_id) {
            const IVF& list = index.lists[list_id];
            if (list.list_size != 0 &&
                (!list.sqrt_candidate2centroid || !list.candidate2centroid)) {
                throw std::runtime_error(
                    "Triangle+PPD requested but the IVF index has no centroid radii");
            }
        }
    }
    init_result(MetricType::METRIC_L2, query_count * k, distances, labels);
    PpdSearchStats total;
    const auto total_start = std::chrono::steady_clock::now();

#pragma omp parallel
    {
        PpdSearchStats local;
        std::vector<float> projected(index.d);
        std::vector<std::pair<float, size_t>> coarse(index.nlist);
#pragma omp for schedule(static)
        for (size_t query_id = 0; query_id < query_count; ++query_id) {
            const float* query = queries + query_id * index.d;
            float* result_distances = distances + query_id * k;
            idx_t* result_labels = labels + query_id * k;

            auto begin = std::chrono::steady_clock::now();
            ppd.transform_query(query, projected.data());
            local.projection_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

            begin = std::chrono::steady_clock::now();
            for (size_t list_id = 0; list_id < index.nlist; ++list_id) {
                coarse[list_id] = {
                    calculatedEuclideanDistance(
                        query, index.centroid_codes.get() + list_id * index.d, index.d),
                    list_id};
            }
            std::partial_sort(
                coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(nprobe), coarse.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            local.coarse_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

            begin = std::chrono::steady_clock::now();
            for (size_t probe = 0; probe < nprobe; ++probe) {
                const size_t list_id = coarse[probe].second;
                const float query_to_centroid2 = coarse[probe].first;
                const IVF& list = index.lists[list_id];
                local.candidates += list.list_size;
                size_t scan_begin = 0;
                size_t scan_end = list.list_size;
                if (triangle) {
                    const float radius = std::sqrt(result_distances[0]);
                    const float query_radius = std::sqrt(query_to_centroid2);
                    while (scan_begin < scan_end &&
                           radius + list.sqrt_candidate2centroid[scan_begin] < query_radius) {
                        ++scan_begin;
                    }
                    const float upper = (radius + query_radius) * (radius + query_radius);
                    while (scan_end > scan_begin &&
                           list.candidate2centroid[scan_end - 1] > upper) {
                        --scan_end;
                    }
                    local.triangle_pruned += list.list_size - (scan_end - scan_begin);
                }
                const float* transformed = ppd.transformed_lists[list_id].data();
                for (size_t candidate = scan_begin; candidate < scan_end; ++candidate) {
                    ++local.ppd_checked;
                    const float* code = transformed + candidate * index.d;
                    float partial = 0.0f;
                    bool pruned = false;
                    for (size_t block = 0; block < index.d; block += block_size) {
                        const size_t end = std::min(index.d, block + block_size);
                        float block_sum = 0.0f;
#pragma omp simd reduction(+ : block_sum)
                        for (size_t axis = block; axis < end; ++axis) {
                            const float delta = projected[axis] - code[axis];
                            block_sum += delta * delta;
                        }
                        partial += block_sum;
                        local.dimensions += end - block;
                        // Small relative slack prevents a roundoff-only false prune.
                        const float threshold = result_distances[0];
                        if (end < index.d &&
                            partial > threshold + 1e-5f * std::max(1.0f, threshold)) {
                            pruned = true;
                            ++local.ppd_pruned;
                            break;
                        }
                    }
                    if (!pruned) {
                        ++local.full_distances;
                        if (partial < result_distances[0]) {
                            heap_replace_top<MetricType::METRIC_L2>(
                                k, result_distances, result_labels, partial,
                                static_cast<idx_t>(list.candidate_id[candidate]));
                        }
                    }
                }
            }
            local.verification_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            heap_sort<MetricType::METRIC_L2>(k, result_distances, result_labels);
        }
#pragma omp critical
        {
            total.candidates += local.candidates;
            total.triangle_pruned += local.triangle_pruned;
            total.ppd_checked += local.ppd_checked;
            total.ppd_pruned += local.ppd_pruned;
            total.full_distances += local.full_distances;
            total.dimensions += local.dimensions;
            total.projection_seconds += local.projection_seconds;
            total.coarse_seconds += local.coarse_seconds;
            total.verification_seconds += local.verification_seconds;
        }
    }
    total.total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
    // Stage counters are summed per-query time. Divide by query_count for a
    // directly comparable average latency; total_seconds is end-to-end wall time.
    return total;
}

}  // namespace tribase
