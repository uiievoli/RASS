#pragma once
#include <array>
#include <bit>
#include <chrono>
#include <limits>

#include "MultiPivot.h"
#include "common.h"
#include "heap.hpp"
#include "stats.h"
#include "utils.h"

#define MANUAL_SIMD

#ifndef MANUAL_SIMD
#include "hnswlib/hnswlib.h"
#endif

namespace tribase {

class IVFScanBase {
   public:
    size_t d;
    size_t k;
    const float* query;
    float query_norm;
    // Result of the most recent list scan. Kept independently of optional
    // profiling counters so release builds can drive list-level pivot choice.
    size_t last_multipivot_checks = 0;
    size_t last_multipivot_pruned = 0;

    IVFScanBase(size_t d, size_t k)
        : d(d), k(k) {}

    void set_query(const float* query) {
        this->query = query;
        this->query_norm = calculatedInnerProduct(query, query, d);
    }

    virtual void lite_scan_codes(size_t list_size,
                                 const float* codes,
                                 const size_t* ids,
                                 float* simi,
                                 idx_t* idxi) = 0;

    virtual void scan_codes(size_t scan_begin,
                            size_t scan_end,
                            size_t list_size,
                            const float* codes,
                            const size_t* ids,
                            float* simi,
                            idx_t* idxi) = 0;

    virtual void scan_codes(size_t scan_begin,
                            size_t scan_end,
                            size_t list_size,
                            const float* codes,
                            const size_t* ids,
                            const float* codes_norms,
                            const float centroid2query,
                            const float* candicate2centroid,
                            const float* sqrt_candicate2centroid,
                            const size_t sub_k,
                            const idx_t* nearest_IP_id,
                            const float* nearest_IP_dis,
                            const idx_t* farest_IP_id,
                            const float* farest_IP_dis,
                            const idx_t* nearest_L2_id,
                            const float* nearest_L2_dis,
                            const PivotMetadata* pivot_metadata,
                            const double* query_pivot_squared,
                            size_t active_pivot_count,
                            MultiPivotMode multipivot_mode,
                            float* query_signature_z,
                            float query_sqrt_rho,
                            size_t projection_prefix_length,
                            size_t projection_block_size,
                            bool projection_dynamic,
                            size_t* projection_ready_dims,
                            bool query_signature_ready,
                            bool* if_skip,
                            float* simi,
                            idx_t* idxi,
                            Stats* stats,
                            const float* centroid_code,
                            float sqrt_ratio,
                            float i_ratio) = 0;
};

template <MetricType metric, OptLevel opt_level, EdgeDevice edge_device_enabled>
class IVFScan : public IVFScanBase {
   public:

#if defined(MANUAL_SIMD)
    using dis_calculator_t = std::function<float(const float*, const float*, size_t)>;
    dis_calculator_t dis_calculator;
    IVFScan(size_t d, size_t k)
        : IVFScanBase(d, k) {
        if constexpr (metric == MetricType::METRIC_IP) {
            if constexpr (edge_device_enabled) {
                dis_calculator = calculatedInnerProduct0;
            } else {
                dis_calculator = calculatedInnerProduct;
            }
        } else if constexpr (metric == MetricType::METRIC_L2) {
            if constexpr (edge_device_enabled) {
                dis_calculator = calculatedEuclideanDistance0;
            } else {
                dis_calculator = calculatedEuclideanDistance;
            }
        } else {
            static_assert(false, "Unsupported metric type");
        }
    }
#else
    hnswlib::DISTFUNC<float> dis_calculator;
    IVFScan(size_t d, size_t k)
        : IVFScanBase(d, k) {
        if constexpr (metric == MetricType::METRIC_IP) {
            auto s = hnswlib::InnerProductSpace(d);
            dis_calculator = s.get_dist_func();
        } else if constexpr (metric == MetricType::METRIC_L2) {
            auto s = hnswlib::L2Space(d);
            dis_calculator = s.get_dist_func();
        } else {
            static_assert(false, "Unsupported metric type");
        }
    }
#endif

    void lite_scan_codes(size_t list_size,
                         const float* codes,
                         const size_t* ids,
                         float* simi,
                         idx_t* idxi) override {
        for (size_t i = 0; i < list_size; i++) {
            const float* candicate = codes + i * d;
            float dis = 0;
            if constexpr (metric == MetricType::METRIC_IP) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedInnerProduct(query, candicate, d);
                } else {
                    dis = calculatedInnerProduct0(query, candicate, d);
                }
                if (dis > simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else if constexpr (metric == MetricType::METRIC_L2) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedEuclideanDistance(query, candicate, d);
                } else {
                    dis = calculatedEuclideanDistance0(query, candicate, d);
                }
                if (dis < simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else {
                static_assert(false, "Unsupported metric type");
            }
        }
    }

    void scan_codes(size_t scan_begin,
                    size_t scan_end,
                    [[maybe_unused]] size_t list_size,
                    const float* codes,
                    const size_t* ids,
                    float* simi,
                    idx_t* idxi) override {
        for (size_t i = scan_begin; i < scan_end; i++) {
            const float* candicate = codes + i * d;
            float dis = 0;
            if constexpr (metric == MetricType::METRIC_IP) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedInnerProduct(query, candicate, d);
                } else {
                    dis = calculatedInnerProduct0(query, candicate, d);
                }
                if (dis > simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else if constexpr (metric == MetricType::METRIC_L2) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedEuclideanDistance(query, candicate, d);
                } else {
                    dis = calculatedEuclideanDistance0(query, candicate, d);
                }
                if (dis < simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else {
                static_assert(false, "Unsupported metric type");
            }
        }
    }

    void scan_codes(size_t scan_begin,
                    size_t scan_end,
                    [[maybe_unused]] size_t list_size,
                    const float* codes,
                    const size_t* ids,
                    [[maybe_unused]] const float* codes_norms,
                    const float centroid2query,
                    const float* candicate2centroid,
                    const float* sqrt_candicate2centroid,
                    const size_t sub_k,
                    const idx_t* nearest_IP_id,
                    const float* nearest_IP_dis,
                    const idx_t* farest_IP_id,
                    const float* farest_IP_dis,
                    const idx_t* nearest_L2_id,
                    const float* nearest_L2_dis,
                    const PivotMetadata* pivot_metadata,
                    const double* query_pivot_squared,
                    size_t active_pivot_count,
                    MultiPivotMode multipivot_mode,
                    float* query_signature_z,
                    float query_sqrt_rho,
                    size_t projection_prefix_length,
                    size_t projection_block_size,
                    bool projection_dynamic,
                    size_t* projection_ready_dims,
                    bool query_signature_ready,
                    bool* if_skip,
                    float* simi,
                    idx_t* idxi,
                    Stats* stats,
                    [[maybe_unused]] const float* centroid_code = nullptr,
                    float sqrt_ratio = 1,
                    float i_ratio = 1) override {
        last_multipivot_checks = 0;
        last_multipivot_pruned = 0;
        float max_radius;
        float diff_cos, diff_sin;
        float max_radius_plus_centroid2query;
        float inv_two_times_sqrt_max_radius_times_centroid2query;
        float inv_sqrt_centroid2query;
        float point5_times_inv_sqrt_centroid2query;
        float sqrt_simi;

        // auto dis_calculator = [](const float* vec1, const float* vec2, size_t size) {
        //     if constexpr (metric == MetricType::METRIC_IP) {
        //         return calculatedInnerProduct(vec1, vec2, size);
        //     } else if constexpr (metric == MetricType::METRIC_L2) {
        //         return calculatedEuclideanDistance(vec1, vec2, size);
        //     } else {
        //         static_assert(false, "Unsupported metric type");
        //     }
        // };

        auto dis_comparator = [](float dis, float simi) {
            if constexpr (metric == MetricType::METRIC_IP) {
                return dis > simi;
            } else if constexpr (metric == MetricType::METRIC_L2) {
                return dis < simi;
            } else {
                static_assert(false, "Unsupported metric type");
            }
        };

        if constexpr (opt_level & OptLevel::OPT_SUBNN_IP) {
            max_radius = candicate2centroid[scan_end - 1];
            max_radius_plus_centroid2query = max_radius + centroid2query;
            inv_two_times_sqrt_max_radius_times_centroid2query = 1 / (2 * sqrt(max_radius * centroid2query));
            inv_sqrt_centroid2query = 1 / sqrt(centroid2query);
            point5_times_inv_sqrt_centroid2query = 0.5 * inv_sqrt_centroid2query;
            if (max_radius + simi[0] >= centroid2query) {
                diff_cos = sqrt(centroid2query - simi[0]) * inv_sqrt_centroid2query;
            } else {
                diff_cos =
                    (max_radius_plus_centroid2query - simi[0]) * inv_two_times_sqrt_max_radius_times_centroid2query;
            }
            diff_sin = sqrt(1 - diff_cos * diff_cos);
        }
        if constexpr (opt_level & OptLevel::OPT_SUBNN_L2) {
            sqrt_simi = sqrt(simi[0]);
        }

        // List-level multipivot validation once (not per candidate).
        bool list_metadata_ok = false;
        bool list_signature_ok = false;
        bool list_degenerate = false;
        bool staged_projection_ok = false;
        float query_prefix_sqrt_rho = 0.0f;
        if constexpr (metric == MetricType::METRIC_L2) {
            if (multipivot_mode != MultiPivotMode::NONE) {
                list_metadata_ok =
                    pivot_metadata != nullptr && query_pivot_squared != nullptr &&
                    active_pivot_count != 0 && active_pivot_count <= pivot_metadata->count &&
                    pivot_metadata->candidate_centroid_squared.size() == list_size;
                if (multipivot_mode == MultiPivotMode::PROJECTION) {
                    list_degenerate =
                        list_metadata_ok && pivot_metadata->fallback_count != 0;
                    list_signature_ok =
                        list_metadata_ok && query_signature_ready && query_signature_z != nullptr &&
                        pivot_metadata->usable_signature(active_pivot_count) &&
                        pivot_metadata->signature_candidate_count == list_size &&
                        (projection_prefix_length == 0 ||
                         (pivot_metadata->candidate_prefix_sqrt_rho.size() == list_size &&
                          pivot_metadata->candidate_prefix_length == std::min(
                              projection_prefix_length, active_pivot_count - 1)));
                    staged_projection_ok =
                        list_signature_ok && projection_prefix_length != 0 &&
                        pivot_metadata->signature_precision == SignaturePrecision::FLOAT32 &&
                        !pivot_metadata->signature_data.empty();
                    if (staged_projection_ok) {
                        float remaining = static_cast<float>(query_pivot_squared[0]);
                        const size_t prefix_count = std::min(
                            projection_prefix_length, active_pivot_count - 1);
                        for (size_t dim = 0; dim < prefix_count; ++dim) {
                            remaining = std::max(
                                0.0f, remaining - query_signature_z[dim] * query_signature_z[dim]);
                        }
                        query_prefix_sqrt_rho = std::sqrt(remaining);
                    }
                }
            }
        }

        // Hot path used by the current PCA experiments. Generate one SIMD lower-bound
        // mask for eight candidates, then enumerate only surviving lanes. Keeping this
        // outside the generic per-candidate loop hoists mode/metadata checks and gives
        // accepted vector prefetches useful lead time before their exact L2 evaluation.
        if constexpr (metric == MetricType::METRIC_L2 &&
                      !(opt_level & OptLevel::OPT_SUBNN_IP) &&
                      !(opt_level & OptLevel::OPT_SUBNN_L2)) {
            const bool use_float32_projection_batch =
                multipivot_mode == MultiPivotMode::PROJECTION && list_signature_ok &&
                !list_degenerate &&
                pivot_metadata->signature_precision == SignaturePrecision::FLOAT32 &&
                !pivot_metadata->signature_data.empty();
            if (use_float32_projection_batch) {
                const float* signature_soa = pivot_metadata->signature_data.data();
                const size_t signature_count = pivot_metadata->signature_candidate_count;
                // A PCA index built at Pmax stores the full residual only in its
                // final SoA row.  For a smaller active prefix, reconstruct the
                // residual from ||x-c||^2 and the selected z coordinates.
                const bool active_prefix = active_pivot_count < pivot_metadata->count;
                const size_t active_prefix_block = active_pivot_count > 1
                    ? active_pivot_count - 1 : 0;

                auto evaluate_survivor = [&](size_t candidate_index) {
#ifdef TRIBASE_ENABLE_STATS
                    const auto exact_start = std::chrono::steady_clock::now();
#endif
                    const float* candidate = codes + candidate_index * d;
                    const float dis = dis_calculator(query, candidate, d);
                    IF_STATS { stats->candidate_distance_computations++; }
                    if (dis_comparator(dis, simi[0])) [[unlikely]] {
                        IF_STATS { stats->simi_update_count++; }
                        heap_replace_top<metric>(k, simi, idxi, dis, ids[candidate_index]);
                    }
#ifdef TRIBASE_ENABLE_STATS
                    if (stats != nullptr) {
                        stats->candidate_exact_seconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - exact_start)
                                .count();
                    }
#endif
                };

                // Dynamic mode chooses one prefix per list. Materialize that query
                // prefix once, evaluate one SIMD lower bound per candidate, then send
                // only survivors to the exact-distance kernel. There is deliberately
                // no multi-stage active-candidate maintenance here.
                if (projection_dynamic && projection_block_size != 0 &&
                    !staged_projection_ok && active_pivot_count != 0) {
                    const size_t projected_dims = active_pivot_count - 1;
#ifdef TRIBASE_ENABLE_STATS
                    const auto projection_start = std::chrono::steady_clock::now();
#endif
                    for (size_t dim = 0; dim < projected_dims; ++dim) {
                        float qz = query_signature_z[dim];
                        if (projection_ready_dims == nullptr || dim >= *projection_ready_dims) {
                            qz = 0.0f;
                            const float* center = pivot_metadata->codes.data();
#pragma omp simd reduction(+ : qz)
                            for (size_t axis = 0; axis < d; ++axis) {
                                qz += (query[axis] - center[axis]) *
                                        pivot_metadata->U[dim * d + axis];
                            }
                            query_signature_z[dim] = qz;
                        }
                    }
                    if (projection_ready_dims != nullptr) {
                        *projection_ready_dims = std::max(*projection_ready_dims, projected_dims);
                    }
                    // The query residual depends only on this list and its chosen
                    // prefix.  Compute it once instead of rebuilding it in every
                    // eight-candidate lower-bound batch.
                    float dynamic_query_residual_squared =
                        static_cast<float>(query_pivot_squared[0]);
                    for (size_t dim = 0; dim < projected_dims; ++dim) {
                        dynamic_query_residual_squared = std::max(
                            0.0f,
                            dynamic_query_residual_squared -
                                query_signature_z[dim] * query_signature_z[dim]);
                    }
                    const float dynamic_query_sqrt_rho =
                        std::sqrt(dynamic_query_residual_squared);
#ifdef TRIBASE_ENABLE_STATS
                    if (stats != nullptr) {
                        stats->candidate_projection_seconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - projection_start)
                                .count();
                    }
                    const auto lb_scan_start = std::chrono::steady_clock::now();
                    const double exact_before =
                        stats != nullptr ? stats->candidate_exact_seconds : 0.0;
#endif

                    size_t base = scan_begin;
                    for (; base + 8 <= scan_end; base += 8) {
                        uint8_t eligible = 0xff;
                        if (if_skip != nullptr) {
                            eligible = 0;
                            for (unsigned lane = 0; lane < 8; ++lane) {
                                if (!if_skip[base + lane]) {
                                    eligible |= static_cast<uint8_t>(1u << lane);
                                }
                            }
                            if (eligible == 0) continue;
                        }
                        const uint8_t pruned = static_cast<uint8_t>(
                            blockSignaturePruneMask8(
                                query_signature_z,
                                static_cast<float>(query_pivot_squared[0]),
                                signature_soa,
                                pivot_metadata->candidate_centroid_squared.data(),
                                base, signature_count, active_pivot_count,
                                projected_dims, simi[0], dynamic_query_sqrt_rho) & eligible);
                        last_multipivot_checks += std::popcount(eligible);
                        last_multipivot_pruned += std::popcount(pruned);
                        uint8_t survivors = static_cast<uint8_t>(eligible & ~pruned);
                        IF_STATS {
                            stats->multipivot_checks += std::popcount(eligible);
                            stats->multipivot_pruned += std::popcount(pruned);
                        }
                        while (survivors != 0) {
                            const unsigned lane = std::countr_zero(survivors);
                            evaluate_survivor(base + lane);
                            survivors = static_cast<uint8_t>(survivors & (survivors - 1));
                        }
                    }
                    for (; base < scan_end; ++base) {
                        if (if_skip != nullptr && if_skip[base]) continue;
                        ++last_multipivot_checks;
                        IF_STATS { stats->multipivot_checks++; }
                        if (blockSignaturePrunes(
                                query_signature_z,
                                static_cast<float>(query_pivot_squared[0]),
                                signature_soa,
                                pivot_metadata->candidate_centroid_squared.data(),
                                base, signature_count, active_pivot_count,
                                projected_dims, simi[0], dynamic_query_sqrt_rho)) {
                            ++last_multipivot_pruned;
                            IF_STATS { stats->multipivot_pruned++; }
                            continue;
                        }
                        evaluate_survivor(base);
                    }
#ifdef TRIBASE_ENABLE_STATS
                    if (stats != nullptr) {
                        const double scan_seconds =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - lb_scan_start)
                                .count();
                        const double exact_seconds =
                            stats->candidate_exact_seconds - exact_before;
                        stats->candidate_lb_seconds +=
                            std::max(0.0, scan_seconds - exact_seconds);
                    }
#endif
                    return;
                }

#ifdef TRIBASE_ENABLE_STATS
                const auto lb_scan_start = std::chrono::steady_clock::now();
                const double exact_before =
                    stats != nullptr ? stats->candidate_exact_seconds : 0.0;
#endif
                size_t base = scan_begin;
                for (; base + 8 <= scan_end; base += 8) {
                    uint8_t eligible = 0xff;
                    if (if_skip != nullptr) {
                        eligible = 0;
                        for (unsigned lane = 0; lane < 8; ++lane) {
                            if (!if_skip[base + lane]) {
                                eligible |= static_cast<uint8_t>(1u << lane);
                            }
                        }
                        if (eligible == 0) continue;
                    }

                    uint8_t prefix_pruned_mask = 0;
                    const uint8_t lower_bound_mask = ((projection_block_size != 0 || active_prefix) &&
                                                      !staged_projection_ok)
                        ? blockSignaturePruneMask8(
                              query_signature_z, static_cast<float>(query_pivot_squared[0]),
                              signature_soa,
                              pivot_metadata->candidate_centroid_squared.data(),
                              base, signature_count, active_pivot_count,
                              active_prefix ? active_prefix_block : projection_block_size, simi[0])
                        : staged_projection_ok
                        ? stagedSignaturePruneMask8(
                              query_signature_z, query_sqrt_rho, query_prefix_sqrt_rho,
                              signature_soa, pivot_metadata->candidate_prefix_sqrt_rho.data(),
                              base, signature_count, active_pivot_count,
                              projection_prefix_length, simi[0],
#ifdef TRIBASE_ENABLE_STATS
                              &prefix_pruned_mask
#else
                              nullptr
#endif
                              )
                        : signaturePruneMask8(query_signature_z, query_sqrt_rho, signature_soa,
                                              base, signature_count, active_pivot_count, simi[0]);
                    const uint8_t pruned =
                        static_cast<uint8_t>(lower_bound_mask & eligible);
                    last_multipivot_checks += std::popcount(eligible);
                    last_multipivot_pruned += std::popcount(pruned);
                    uint8_t survivors = static_cast<uint8_t>(eligible & ~pruned);
                    IF_STATS {
                        stats->multipivot_checks += std::popcount(eligible);
                        stats->multipivot_pruned += std::popcount(pruned);
                        stats->multipivot_prefix_pruned +=
                            std::popcount(static_cast<uint8_t>(prefix_pruned_mask & eligible));
                    }

                    // Issue loads for all accepted lanes before starting exact distances.
                    // The second line matters for GloVe25 (100-byte vectors).
                    uint8_t pending = survivors;
                    while (pending != 0) {
                        const unsigned lane = std::countr_zero(pending);
                        const float* candidate = codes + (base + lane) * d;
                        _mm_prefetch(reinterpret_cast<const char*>(candidate), _MM_HINT_T0);
                        if (d > 16) {
                            _mm_prefetch(reinterpret_cast<const char*>(candidate + 16),
                                         _MM_HINT_T0);
                        }
                        pending = static_cast<uint8_t>(pending & (pending - 1));
                    }

                    while (survivors != 0) {
                        const unsigned lane = std::countr_zero(survivors);
                        evaluate_survivor(base + lane);
                        survivors = static_cast<uint8_t>(survivors & (survivors - 1));
                    }
                }

                // At most seven candidates: use the scalar bound without re-entering the
                // generic mode-dispatch loop.
                for (; base < scan_end; ++base) {
                    if (if_skip != nullptr && if_skip[base]) continue;
                    ++last_multipivot_checks;
                    IF_STATS { stats->multipivot_checks++; }
                    bool prefix_pruned = false;
                    const bool pruned = ((projection_block_size != 0 || active_prefix) &&
                                         !staged_projection_ok)
                        ? blockSignaturePrunes(
                              query_signature_z, static_cast<float>(query_pivot_squared[0]),
                              pivot_metadata->signature_data.data(),
                              pivot_metadata->candidate_centroid_squared.data(),
                              base, pivot_metadata->signature_candidate_count, active_pivot_count,
                              active_prefix ? active_prefix_block : projection_block_size, simi[0])
                        : staged_projection_ok
                        ? stagedSignaturePrunes(
                              query_signature_z, query_sqrt_rho, query_prefix_sqrt_rho,
                              signature_soa, pivot_metadata->candidate_prefix_sqrt_rho.data(),
                              base, signature_count, active_pivot_count,
                              projection_prefix_length, simi[0],
#ifdef TRIBASE_ENABLE_STATS
                              &prefix_pruned
#else
                              nullptr
#endif
                              )
                        : signaturePrunes(query_signature_z, query_sqrt_rho, signature_soa, base,
                                          signature_count, active_pivot_count, simi[0]);
                    if (pruned) {
                        ++last_multipivot_pruned;
                        IF_STATS {
                            stats->multipivot_pruned++;
                            if (prefix_pruned) stats->multipivot_prefix_pruned++;
                        }
                        continue;
                    }
                    const float* candidate = codes + base * d;
                    _mm_prefetch(reinterpret_cast<const char*>(candidate), _MM_HINT_T0);
                    if (d > 16) {
                        _mm_prefetch(reinterpret_cast<const char*>(candidate + 16), _MM_HINT_T0);
                    }
                    evaluate_survivor(base);
                }
#ifdef TRIBASE_ENABLE_STATS
                if (stats != nullptr) {
                    const double scan_seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                      lb_scan_start)
                            .count();
                    const double exact_seconds =
                        stats->candidate_exact_seconds - exact_before;
                    stats->candidate_lb_seconds +=
                        std::max(0.0, scan_seconds - exact_seconds);
                }
#endif
                return;
            }
        }

        // AVX2 batch prune mask (8 candidates). Recomputed when the window moves;
        // a stale larger radius only under-prunes (still correct).
        uint8_t pending_prune_mask = 0;
        size_t pending_prune_base = std::numeric_limits<size_t>::max();

        for (size_t i = scan_begin; i < scan_end; i++) {
            if (if_skip && if_skip[i]) {
                continue;
            }
            if constexpr ((opt_level & OptLevel::OPT_SUBNN_IP) || (opt_level & OptLevel::OPT_SUBNN_L2)) {
                _mm_prefetch((char*)(if_skip + i + 1), _MM_HINT_T0);
            }
            const float* candicate = codes + i * d;
            float dis;
            if constexpr (metric == MetricType::METRIC_L2) {
                if (multipivot_mode != MultiPivotMode::NONE) {
                    IF_STATS { stats->multipivot_checks++; }
                    bool can_prune = false;
                    bool should_skip = false;
                    if (multipivot_mode == MultiPivotMode::MAX) {
                        if (list_metadata_ok) {
                            const double* candidate_pivot_squared =
                                pivot_metadata->candidate_pivot_squared.data() + i * pivot_metadata->count;
                            const double lower_bound_squared = maxSinglePivotLowerBoundSquared(
                                query_pivot_squared, candidate_pivot_squared, active_pivot_count);
                            can_prune = true;
                            should_skip = lower_bound_squared > static_cast<double>(simi[0]);
                        }
                    } else if (list_signature_ok && !list_degenerate) {
                        can_prune = true;
                        if constexpr (!(opt_level & OptLevel::OPT_SUBNN_IP) &&
                                      !(opt_level & OptLevel::OPT_SUBNN_L2)) {
                            if (i + 8 <= scan_end &&
                                (pending_prune_base == std::numeric_limits<size_t>::max() ||
                                 i >= pending_prune_base + 8)) {
                                pending_prune_mask = staged_projection_ok
                                    ? stagedSignaturePruneMask8(
                                          query_signature_z, query_sqrt_rho,
                                          query_prefix_sqrt_rho,
                                          pivot_metadata->signature_data.data(),
                                          pivot_metadata->candidate_prefix_sqrt_rho.data(), i,
                                          pivot_metadata->signature_candidate_count,
                                          active_pivot_count, projection_prefix_length, simi[0])
                                    : signaturePruneMask8(
                                          query_signature_z, query_sqrt_rho, *pivot_metadata, i,
                                          active_pivot_count, simi[0]);
                                pending_prune_base = i;
                            }
                            if (i >= pending_prune_base && i < pending_prune_base + 8) {
                                should_skip =
                                    (pending_prune_mask &
                                     static_cast<uint8_t>(1u << (i - pending_prune_base))) != 0;
                            } else {
                                should_skip = staged_projection_ok
                                    ? stagedSignaturePrunes(
                                          query_signature_z, query_sqrt_rho,
                                          query_prefix_sqrt_rho,
                                          pivot_metadata->signature_data.data(),
                                          pivot_metadata->candidate_prefix_sqrt_rho.data(), i,
                                          pivot_metadata->signature_candidate_count,
                                          active_pivot_count, projection_prefix_length, simi[0])
                                    : signaturePrunes(
                                          query_signature_z, query_sqrt_rho, *pivot_metadata, i,
                                          active_pivot_count, simi[0]);
                            }
                        } else {
                            should_skip = staged_projection_ok
                                ? stagedSignaturePrunes(
                                      query_signature_z, query_sqrt_rho,
                                      query_prefix_sqrt_rho,
                                      pivot_metadata->signature_data.data(),
                                      pivot_metadata->candidate_prefix_sqrt_rho.data(), i,
                                      pivot_metadata->signature_candidate_count,
                                      active_pivot_count, projection_prefix_length, simi[0])
                                : signaturePrunes(
                                      query_signature_z, query_sqrt_rho, *pivot_metadata, i,
                                      active_pivot_count, simi[0]);
                        }
                    }
                    IF_STATS {
                        if (!can_prune) stats->multipivot_invalid++;
                    }
                    // simi[0] is the current squared L2 heap radius.
                    if (can_prune && should_skip) {
                        IF_STATS { stats->multipivot_pruned++; }
                        continue;
                    }
                }
            }
            // Keep the baseline look-ahead, but never prefetch an MP candidate before its
            // signature is accepted. Sparse survivors otherwise pull pruned vectors into L1.
            if (multipivot_mode == MultiPivotMode::NONE) {
                _mm_prefetch((char*)(codes + (i + 1) * d), _MM_HINT_T0);
            } else {
                _mm_prefetch((char*)candicate, _MM_HINT_T0);
            }
            if constexpr (metric == MetricType::METRIC_L2) {
#ifndef MANUAL_SIMD
                dis = dis_calculator(query, candicate, &d);
#else
                dis = dis_calculator(query, candicate, d);
#endif
            } else {
#ifndef MANUAL_SIMD
                dis = dis_calculator(query, candicate, &d);
#else
                dis = dis_calculator(query, candicate, d);
#endif
            }
            IF_STATS {
                stats->candidate_distance_computations++;
            }

            if (dis_comparator(dis, simi[0])) [[unlikely]] {
                IF_STATS {
                    stats->simi_update_count++;
                }
                idx_t id = ids[i];
                heap_replace_top<metric>(k, simi, idxi, dis, id);

                if constexpr (opt_level & OptLevel::OPT_SUBNN_IP) {
                    if (max_radius + simi[0] >= centroid2query) {
                        diff_cos = sqrt(centroid2query - simi[0]) * inv_sqrt_centroid2query;
                    } else {
                        diff_cos = (max_radius_plus_centroid2query - simi[0]) *
                                   inv_two_times_sqrt_max_radius_times_centroid2query;
                    }
                    diff_sin = sqrt(1 - diff_cos * diff_cos);
                }

                if constexpr (opt_level & OptLevel::OPT_SUBNN_L2) {
                    sqrt_simi = sqrt(simi[0]);
                }
            }

            if constexpr (opt_level & OptLevel::OPT_SUBNN_L2) {
                IF_STATS {
                    stats->check_subnn_L2_count += 1;
                }
                size_t skip_fake_id_begin = i * sub_k;
                size_t skip_fake_id_end = skip_fake_id_begin + sub_k;
                skip_fake_id_begin += 1;
                for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end; skip_fake_id++) {
                    float tmp_plus = nearest_L2_dis[skip_fake_id] + sqrt_simi;
                    int64_t skip_true_id = nearest_L2_id[skip_fake_id];
                    if (skip_true_id >= 0 && dis > sqrt_ratio * tmp_plus * tmp_plus) {  // already sqrt nearest_L2_dis
#ifdef CORRECTNESS_CHECK
#ifndef MANUAL_SIMD
                        float true_dis = dis_calculator(query, codes + skip_true_id * d, &d);
#else
                        float true_dis = dis_calculator(query, codes + skip_true_id * d, d);
#endif
#pragma omp critical
                        if (true_dis < simi[0]) {
                            std::cerr << "Error: " << true_dis << " " << dis << " " << nearest_L2_id[skip_fake_id] << std::endl;
                            throw std::runtime_error("SUBNN_L2_NEAREST_ERROR");
                        }
#endif
                        IF_STATS {
                            if (!if_skip[skip_true_id] && i < skip_true_id) { //  
                                // std::ofstream("logs/fuck.txt", std::ios::app) << skip_true_id << " " << skip_fake_id << " " << nearest_L2_dis[skip_fake_id] << " " << sqrt_simi << " " << dis << ", ";
                                stats->skip_subnn_L2_count++;
                            }
                        }
                        if_skip[skip_true_id] = true;
                    } else {
                        // std::ofstream("logs/fuck.txt", std::ios::app) << "["  << skip_true_id << " " << skip_fake_id << " " << nearest_L2_dis[skip_fake_id] << " " << sqrt_simi << " " << dis << " END]";
                        IF_STATS {
                            stats->check_subnn_L2_ele_count += skip_fake_id - skip_fake_id_begin;
                        }
                        break;
                    }
                }
                // std::ofstream("logs/fuck.txt", std::ios::app) << " " << skip_fake_id_end << std::endl;
            }

            if constexpr (opt_level & OptLevel::OPT_SUBNN_IP) {
                if (centroid2query > simi[0]) {
                    float this_cos = (candicate2centroid[i] + centroid2query - dis) *
                                     point5_times_inv_sqrt_centroid2query / sqrt_candicate2centroid[i];
                    float this_sin = sqrt(1 - this_cos * this_cos);

                    float tmpa = diff_cos * this_cos;
                    float tmpb = diff_sin * this_sin;
                    float cut_degree_cos_minus = tmpa + tmpb;
                    float cut_degree_cos_plus = tmpa - tmpb;

                    if (this_cos < diff_cos && this_cos > -diff_cos) {
                        IF_STATS {
                            stats->check_subnn_IP_count += 2;
                        }
                        size_t skip_fake_id_begin = i * sub_k;
                        size_t skip_fake_id_end = skip_fake_id_begin + sub_k;
                        skip_fake_id_begin += 1;
                        for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end;
                             skip_fake_id++) {
                            int64_t skip_true_id = nearest_IP_id[skip_fake_id];
                            if (skip_true_id >= 0 && nearest_IP_dis[skip_fake_id] > i_ratio * cut_degree_cos_minus) {
#ifdef CORRECTNESS_CHECK
#ifndef MANUAL_SIMD
                                float true_dis = dis_calculator(query, codes + skip_true_id * d, &d);
#else
                                float true_dis = dis_calculator(query, codes + skip_true_id * d, d);
#endif
#pragma omp critical
                                if (true_dis < simi[0]) {
                                    std::cerr << std::format("Error: query->p2: {} <= {}, nearestesIP: {}, cut: {}", sqrt(true_dis), sqrt(simi[0]), nearest_IP_dis[skip_fake_id], cut_degree_cos_minus) << std::endl;
                                    std::cerr << std::format("query->p1: {}, max_r: {}, query->c: {}", sqrt(dis), sqrt(max_radius), sqrt(centroid2query)) << std::endl;
                                    std::cerr << std::format("c->p1: {}, c->p2: {}", sqrt(candicate2centroid[i]), sqrt(candicate2centroid[skip_true_id])) << std::endl;
#ifndef MANUAL_SIMD
                                    assert(candicate2centroid[i] == dis_calculator(codes + i * d, centroid_code, &d));
                                    assert(candicate2centroid[skip_true_id] == dis_calculator(codes + skip_true_id * d, centroid_code, &d));
#else
                                    assert(candicate2centroid[i] == dis_calculator(codes + i * d, centroid_code, d));
                                    assert(candicate2centroid[skip_true_id] == dis_calculator(codes + skip_true_id * d, centroid_code, d));
#endif
                                    std::cout << (metric == MetricType::METRIC_L2) << std::endl;

                                    output_codes(centroid_code, d);
                                    output_codes(query, d);
                                    output_codes(codes + i * d, d);
                                    output_codes(codes + skip_true_id * d, d);
                                    // throw std::runtime_error("SUBNN_IP_NEAREST_ERROR");
                                    assert(false);
                                }
#endif
                                IF_STATS {
                                    if (!if_skip[skip_true_id] && i < skip_true_id) {
                                        stats->skip_subnn_IP_count++;
                                    }
                                }
                                if_skip[skip_true_id] = true;
                            } else {
                                IF_STATS {
                                    stats->check_subnn_IP_ele_count += skip_fake_id - skip_fake_id_begin;
                                }
                                break;
                            }
                        }

                        skip_fake_id_begin = i * sub_k;
                        skip_fake_id_end = skip_fake_id_begin + sub_k;
                        for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end;
                             skip_fake_id++) {
                            size_t skip_true_id = farest_IP_id[skip_fake_id];
                            if (skip_true_id > 0 && farest_IP_dis[skip_fake_id] < cut_degree_cos_plus) {
#ifdef CORRECTNESS_CHECK
                                float true_dis = dis_calculator(query, codes + skip_true_id * d, &d);
#pragma omp critical
                                if (true_dis < simi[0]) {
                                    std::cerr << std::format("Error: query->p2: {} <= {}, farestesIP: {}, cut: {}", sqrt(true_dis), sqrt(simi[0]), farest_IP_dis[skip_fake_id], cut_degree_cos_plus) << std::endl;
                                    std::cerr << std::format("query->p1: {}, max_r: {}, query->c: {}", sqrt(dis), sqrt(max_radius), sqrt(centroid2query)) << std::endl;
                                    std::cerr << std::format("c->p1: {}, c->p2: {}", sqrt(candicate2centroid[i]), sqrt(candicate2centroid[skip_true_id])) << std::endl;
                                    output_codes(centroid_code, d);
                                    output_codes(query, d);
                                    output_codes(codes + i * d, d);
                                    output_codes(codes + skip_true_id * d, d);
                                    // throw std::runtime_error("SUBNN_IP_FAREST_ERROR");
                                    assert(false);
                                }
#endif
                                IF_STATS {
                                    if (!if_skip[skip_true_id] && i < skip_true_id) {
                                        stats->skip_subnn_IP_count++;
                                    }
                                }
                                if_skip[skip_true_id] = true;
                            } else {
                                IF_STATS {
                                    stats->check_subnn_IP_ele_count += skip_fake_id - skip_fake_id_begin;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    };
};  // namespace tribase
}  // namespace tribase
