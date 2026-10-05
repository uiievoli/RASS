#pragma once
#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <iostream>
#include <string_view>
#include <vector>

#include "common.h"
#include "utils.h"

namespace tribase {

// One probed inverted-list visit (per query × nprobe).
struct ListVisitStat {
    size_t query_id = 0;
    size_t probe_rank = 0;  // 0 .. nprobe-1 (closest centroid first)
    idx_t list_id = 0;
    size_t list_size = 0;
    float centroid2query = 0;
    size_t scan_begin = 0;
    size_t scan_end = 0;
    size_t tri = 0;
    size_t tri_large = 0;
    size_t subnn_L2 = 0;
    size_t subnn_IP = 0;
    size_t multipivot_checks = 0;
    size_t multipivot_pruned = 0;
    size_t multipivot_invalid = 0;
    size_t active_pivots = 0;
    size_t candidate_distances = 0;
    size_t pivot_distances = 0;
    size_t simi_update = 0;

    size_t scanned() const { return scan_end > scan_begin ? scan_end - scan_begin : 0; }
    // Vectors that still need a distance after triangle + subnn pruning on this list.
    size_t dis_calculate() const {
        return candidate_distances;
    }
};

inline void listVisitsToCsv(const std::string& filename,
                            const std::vector<ListVisitStat>& visits,
                            bool append,
                            const std::string& dataset,
                            size_t nlist,
                            size_t nprobe,
                            OptLevel opt_level,
                            MultiPivotMode multipivot_mode,
                            MultiPivotScope multipivot_scope,
                            const std::string& multipivot_method,
                            size_t pivot_count,
                            uint64_t pivot_seed,
                            float simi_ratio) {
    CsvWriter writer(filename,
                     {"dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
                      "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed", "simi_ratio",
                      "query_id", "probe_rank", "list_id", "list_size", "centroid2query",
                      "scan_begin", "scan_end", "scanned",
                      "tri", "tri_large", "subnn_L2", "subnn_IP",
                      "multipivot_checks", "multipivot_pruned", "multipivot_invalid",
                      "active_pivots", "candidate_distances", "pivot_distances",
                      "simi_update", "dis_calculate"},
                     append, false);
    for (const auto& v : visits) {
        writer << dataset << nlist << nprobe << static_cast<int>(opt_level)
               << multiPivotModeName(multipivot_mode) << multiPivotScopeName(multipivot_scope)
               << multipivot_method << pivot_count << pivot_seed << simi_ratio
               << v.query_id << v.probe_rank << v.list_id << v.list_size << v.centroid2query
               << v.scan_begin << v.scan_end << v.scanned()
               << v.tri << v.tri_large << v.subnn_L2 << v.subnn_IP
               << v.multipivot_checks << v.multipivot_pruned << v.multipivot_invalid
               << v.active_pivots << v.candidate_distances << v.pivot_distances
               << v.simi_update << v.dis_calculate() << std::endl;
    }
}

class Stats {
   public:
    size_t total_count;

    // triangle part
    size_t skip_triangle_count;
    size_t skip_triangle_large_count;

    // subNN part
    size_t check_subnn_L2_ele_count;
    size_t check_subnn_IP_ele_count;
    size_t check_subnn_L2_count;
    size_t check_subnn_IP_count;
    size_t skip_subnn_L2_count;
    size_t skip_subnn_IP_count;

    size_t simi_update_count;
    size_t dis_calculate_count;
    size_t multipivot_checks;
    size_t multipivot_pruned;
    size_t multipivot_prefix_pruned;
    size_t multipivot_invalid;
    size_t candidate_distance_computations;
    size_t pivot_distance_computations;
    size_t microblock_checks;
    size_t microblock_pruned;          // blocks pruned
    size_t microblock_vectors_pruned;  // vectors covered by pruned blocks
    // Mutually exclusive worker CPU-time decomposition:
    // - query_signature: query-to-center work plus U(q-c), once globally or per probed list;
    // - candidate_decision: scan_codes, including MP decisions, surviving L2 distances and heap updates;
    // - other: coarse centroid search, triangle/microblock setup, sorting and loop overhead.
    // The three stages sum to search_worker_seconds (up to clock precision).
    double query_signature_seconds;
    double candidate_decision_seconds;
    double other_seconds;
    double search_worker_seconds;
    double candidate_projection_seconds;
    double candidate_lb_seconds;
    double candidate_exact_seconds;
    double candidate_active_copy_seconds;
    std::array<size_t, 513> active_pivot_histogram;

    size_t nlist;
    size_t nprobe;

    double faiss_query_time;
    double query_time;

    OptLevel opt_level;
    MultiPivotMode multipivot_mode = MultiPivotMode::NONE;
    MultiPivotScope multipivot_scope = MultiPivotScope::GLOBAL;
    std::string multipivot_method = "affine_fps";
    size_t pivot_count = 0;
    uint64_t pivot_seed = 0;
    uint32_t index_format_version = INDEX_FORMAT_VERSION;
    std::string signature_precision = "float32";
    size_t projection_prefix_length = 0;

    double recall;
    double r2;
    float simi_ratio;

    size_t n_query;

    // Per-list dump (optional; filled when collect_list_stats is true)
    bool collect_list_stats = false;
    size_t query_offset = 0;
    std::vector<ListVisitStat> list_visits;

    // summary
   private:
    float pruning_triangle;
    float pruning_triangle_large;
    float pruning_subnn_L2;
    float pruning_subnn_IP;
    float check_subnn_L2;
    float check_subnn_IP;

    float simi_update_rate;

    float time_speedup;
    float pruning_speedup;
    double qps;

   public:
    void reset() {
        total_count = 0;
        skip_triangle_count = 0;
        skip_triangle_large_count = 0;
        check_subnn_L2_ele_count = 0;
        check_subnn_IP_ele_count = 0;
        check_subnn_L2_count = 0;
        check_subnn_IP_count = 0;
        skip_subnn_L2_count = 0;
        skip_subnn_IP_count = 0;
        simi_update_count = 0;
        dis_calculate_count = 0;
        multipivot_checks = 0;
        multipivot_pruned = 0;
        multipivot_prefix_pruned = 0;
        multipivot_invalid = 0;
        candidate_distance_computations = 0;
        pivot_distance_computations = 0;
        microblock_checks = 0;
        microblock_pruned = 0;
        microblock_vectors_pruned = 0;
        query_signature_seconds = 0;
        candidate_decision_seconds = 0;
        other_seconds = 0;
        search_worker_seconds = 0;
        candidate_projection_seconds = 0;
        candidate_lb_seconds = 0;
        candidate_exact_seconds = 0;
        candidate_active_copy_seconds = 0;
        active_pivot_histogram.fill(0);
        nlist = 0;
        nprobe = 0;
        faiss_query_time = 0;
        query_time = 0;
        opt_level = OptLevel::OPT_NONE;
        multipivot_mode = MultiPivotMode::NONE;
        multipivot_scope = MultiPivotScope::GLOBAL;
        multipivot_method = "affine_fps";
        pivot_count = 0;
        pivot_seed = 0;
        signature_precision = "float32";
        projection_prefix_length = 0;
        recall = 0;
        r2 = 0;
        simi_ratio = 0;
        n_query = 0;
        list_visits.clear();
    }

    Stats() { reset(); }

    void summary() {
        dis_calculate_count = candidate_distance_computations;
        simi_update_rate = dis_calculate_count == 0 ? 0 : 100.0 * simi_update_count / dis_calculate_count;
        pruning_triangle = skip_triangle_count == 0 ? 0 : 100.0 * skip_triangle_count / total_count;
        pruning_triangle_large = skip_triangle_large_count == 0 ? 0 : 100.0 * skip_triangle_large_count / total_count;
        pruning_subnn_L2 = skip_subnn_L2_count == 0 ? 0 : 100.0 * skip_subnn_L2_count / total_count;
        pruning_subnn_IP = skip_subnn_IP_count == 0 ? 0 : 100.0 * skip_subnn_IP_count / total_count;
        check_subnn_IP = check_subnn_IP_count == 0 ? 0 : 1.0 * check_subnn_IP_ele_count / check_subnn_IP_count;
        check_subnn_L2 = check_subnn_L2_count == 0 ? 0 : 1.0 * check_subnn_L2_ele_count / check_subnn_L2_count;

        time_speedup = query_time == 0 ? 0 : 100.0 * faiss_query_time / query_time;
        pruning_speedup = dis_calculate_count == 0 ? 0 : 100.0 * total_count / dis_calculate_count;
        qps = query_time == 0 ? 0 : static_cast<double>(n_query) / query_time;
        other_seconds = std::max(
            0.0, search_worker_seconds - query_signature_seconds - candidate_decision_seconds);
    }

    void print() {
        summary();
        std::cout                   << std::format("nprobe:{} opt_level: {} multipivot: {} scope: {} P: {} seed: {} sig: {} lb_prefix: {} simi_ratio: {}\n",
                                 nprobe, static_cast<int>(opt_level), multiPivotModeName(multipivot_mode),
                                 multiPivotScopeName(multipivot_scope), pivot_count, pivot_seed,
                                 signature_precision, projection_prefix_length, simi_ratio)
                  << std::format("tri: {}({:.2f}%) tri_large: {}({:.2f}%) subnn_L2: {}({:.2f}%) subnn_IP: {}({:.2f}%)\n", skip_triangle_count, pruning_triangle, skip_triangle_large_count, pruning_triangle_large, skip_subnn_L2_count, pruning_subnn_L2, skip_subnn_IP_count, pruning_subnn_IP)
                  << std::format("simi_update_rate: {:.2f}% check_L2: {} check_IP: {}\n", simi_update_rate, check_subnn_L2, check_subnn_IP)
                  << std::format("multipivot checks: {} pruned: {} prefix_pruned: {} invalid: {} candidate_distances: {} pivot_distances: {}\n",
                                 multipivot_checks, multipivot_pruned, multipivot_prefix_pruned,
                                 multipivot_invalid,
                                 candidate_distance_computations, pivot_distance_computations)
                  << std::format("microblock checks: {} pruned: {} ({:.2f}%) vectors_pruned: {}\n",
                                 microblock_checks, microblock_pruned,
                                 microblock_checks == 0 ? 0.0 : 100.0 * microblock_pruned / microblock_checks,
                                 microblock_vectors_pruned)
                  << std::format(
                         "stage_seconds query_signature: {:.6f} candidate_decision: {:.6f} other: {:.6f} worker_total: {:.6f}\n",
                         query_signature_seconds, candidate_decision_seconds, other_seconds,
                         search_worker_seconds)
                  << std::format(
                         "candidate_detail_seconds projection_state: {:.6f} lb_ub: {:.6f} exact_distance: {:.6f} active_copy: {:.6f}\n",
                         candidate_projection_seconds, candidate_lb_seconds,
                         candidate_exact_seconds, candidate_active_copy_seconds)
                  << "active_pivots:";
        for (size_t p = 0; p < active_pivot_histogram.size(); ++p) {
            if (active_pivot_histogram[p] != 0) {
                std::cout << " " << p << "=" << active_pivot_histogram[p];
            }
        }
        std::cout << "\n"
                  << std::format(
                         "stage_us_per_query query_signature: {:.3f} candidate_decision: {:.3f} other: {:.3f}; shares: {:.2f}%/{:.2f}%/{:.2f}%\n",
                         n_query == 0 ? 0.0 : 1e6 * query_signature_seconds / n_query,
                         n_query == 0 ? 0.0 : 1e6 * candidate_decision_seconds / n_query,
                         n_query == 0 ? 0.0 : 1e6 * other_seconds / n_query,
                         search_worker_seconds == 0 ? 0.0 : 100.0 * query_signature_seconds / search_worker_seconds,
                         search_worker_seconds == 0 ? 0.0 : 100.0 * candidate_decision_seconds / search_worker_seconds,
                         search_worker_seconds == 0 ? 0.0 : 100.0 * other_seconds / search_worker_seconds)
                  << std::format("time_speedup: {:.2f}% pruning_speedup: {:.2f}% faiss_query_time: {:.6f} query_time: {:.6f} qps: {:f}\n", time_speedup, pruning_speedup, faiss_query_time, query_time, qps)
                  << std::format("recall: {} r2: {}\n", recall, r2);
    }

    void toCsv(std::string filename, bool append, std::string dataset = "Unknown") {
        CsvWriter writer(filename,
                         {"dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
                          "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed",
                          "signature_precision", "projection_prefix_length",
                          "index_format_version", "simi_ratio",
                          "tri", "tri_large", "subnn_L2", "subnn_IP", "simi_update_rate",
                          "check_L2", "check_IP",
                          "multipivot_checks", "multipivot_pruned", "multipivot_prefix_pruned",
                          "multipivot_invalid",
                          "candidate_distance_computations", "pivot_distance_computations",
                          "microblock_checks", "microblock_pruned", "microblock_vectors_pruned",
                          "query_signature_seconds", "candidate_decision_seconds", "other_seconds",
                          "candidate_projection_seconds", "candidate_lb_seconds", "candidate_exact_seconds",
                          "candidate_active_copy_seconds",
                          "active_pivots_0", "active_pivots_16", "active_pivots_other", "active_pivots_total",
                          "search_worker_seconds", "query_signature_us_per_query",
                          "candidate_decision_us_per_query", "other_us_per_query",
                          "query_signature_share", "candidate_decision_share", "other_share",
                          "time_speedup", "pruning_speedup", "query_time", "qps",
                          "recall", "r2"},
                         append, false);
        summary();
        writer << dataset << nlist << nprobe << static_cast<int>(opt_level)
               << multiPivotModeName(multipivot_mode) << multiPivotScopeName(multipivot_scope)
               << multipivot_method << pivot_count << pivot_seed << signature_precision
               << projection_prefix_length << index_format_version << simi_ratio
               << skip_triangle_count << skip_triangle_large_count << skip_subnn_L2_count << skip_subnn_IP_count << simi_update_rate / 100
               << check_subnn_L2 << check_subnn_IP
               << multipivot_checks << multipivot_pruned << multipivot_prefix_pruned
               << multipivot_invalid
               << candidate_distance_computations << pivot_distance_computations
               << microblock_checks << microblock_pruned << microblock_vectors_pruned
               << query_signature_seconds << candidate_decision_seconds << other_seconds
               << candidate_projection_seconds << candidate_lb_seconds << candidate_exact_seconds
               << candidate_active_copy_seconds
               << active_pivot_histogram[0] << active_pivot_histogram[16]
               << (std::accumulate(active_pivot_histogram.begin(), active_pivot_histogram.end(), size_t{0})
                   - active_pivot_histogram[0] - active_pivot_histogram[16])
               << std::accumulate(active_pivot_histogram.begin(), active_pivot_histogram.end(), size_t{0})
               << search_worker_seconds
               << (n_query == 0 ? 0.0 : 1e6 * query_signature_seconds / n_query)
               << (n_query == 0 ? 0.0 : 1e6 * candidate_decision_seconds / n_query)
               << (n_query == 0 ? 0.0 : 1e6 * other_seconds / n_query)
               << (search_worker_seconds == 0 ? 0.0 : query_signature_seconds / search_worker_seconds)
               << (search_worker_seconds == 0 ? 0.0 : candidate_decision_seconds / search_worker_seconds)
               << (search_worker_seconds == 0 ? 0.0 : other_seconds / search_worker_seconds)
               << time_speedup / 100 << pruning_speedup / 100 << query_time << qps
               << recall << r2 << std::endl;
    }
};

inline Stats mergeStats(std::vector<Stats>& stats) {
    Stats merged;
    for (auto& s : stats) {
        merged.total_count += s.total_count;
        merged.skip_triangle_count += s.skip_triangle_count;
        merged.skip_triangle_large_count += s.skip_triangle_large_count;
        merged.check_subnn_L2_ele_count += s.check_subnn_L2_ele_count;
        merged.check_subnn_IP_ele_count += s.check_subnn_IP_ele_count;
        merged.check_subnn_L2_count += s.check_subnn_L2_count;
        merged.check_subnn_IP_count += s.check_subnn_IP_count;
        merged.skip_subnn_L2_count += s.skip_subnn_L2_count;
        merged.skip_subnn_IP_count += s.skip_subnn_IP_count;
        merged.simi_update_count += s.simi_update_count;
        merged.multipivot_checks += s.multipivot_checks;
        merged.multipivot_pruned += s.multipivot_pruned;
        merged.multipivot_prefix_pruned += s.multipivot_prefix_pruned;
        merged.multipivot_invalid += s.multipivot_invalid;
        merged.candidate_distance_computations += s.candidate_distance_computations;
        merged.pivot_distance_computations += s.pivot_distance_computations;
        merged.microblock_checks += s.microblock_checks;
        merged.microblock_pruned += s.microblock_pruned;
        merged.microblock_vectors_pruned += s.microblock_vectors_pruned;
        merged.query_signature_seconds += s.query_signature_seconds;
        merged.candidate_decision_seconds += s.candidate_decision_seconds;
        merged.candidate_projection_seconds += s.candidate_projection_seconds;
        merged.candidate_lb_seconds += s.candidate_lb_seconds;
        merged.candidate_exact_seconds += s.candidate_exact_seconds;
        merged.candidate_active_copy_seconds += s.candidate_active_copy_seconds;
        for (size_t p = 0; p < merged.active_pivot_histogram.size(); ++p) {
            merged.active_pivot_histogram[p] += s.active_pivot_histogram[p];
        }
        merged.search_worker_seconds += s.search_worker_seconds;
        if (!s.list_visits.empty()) {
            merged.list_visits.insert(merged.list_visits.end(),
                                      s.list_visits.begin(), s.list_visits.end());
        }
        merged.collect_list_stats = merged.collect_list_stats || s.collect_list_stats;
    }
    return merged;
}
}  // namespace tribase
