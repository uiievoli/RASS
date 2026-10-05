#include "Index.h"

#include "Microblock.h"
#include "common.h"
#include <faiss/Clustering.h>
#include <Eigen/Dense>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>
#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include "IVF.h"
#include "IVFScan.hpp"
#include "heap.hpp"

#define SUB_LIST_SIZE 8ul
#define IP_SUB_RATIO 1.5
#define RECALL_TEST_RATIO 0.1
// #define SUB_STATS

namespace tribase {
namespace {
constexpr char INDEX_MAGIC[8] = {'T', 'R', 'I', 'B', 'A', 'S', 'E', '2'};
constexpr uint32_t INDEX_ENDIAN_MARKER = 0x01020304;

// In global projection mode every list needs only its candidate-side tables.
// The shared basis and all geometry remain in Index::global_pivots. Retaining
// a copy of that geometry in every list wastes memory without serving search.
void discard_global_pivot_geometry(PivotMetadata& metadata) {
    std::vector<idx_t>().swap(metadata.source_ids);
    std::vector<float>().swap(metadata.codes);
    std::vector<double>().swap(metadata.pairwise_squared);
    std::vector<double>().swap(metadata.candidate_pivot_squared);
    std::vector<float>().swap(metadata.candidate_prefix_sqrt_rho);
    std::vector<double>().swap(metadata.gram_pinv);
    std::vector<double>().swap(metadata.gram_eigenvalues);
    std::vector<double>().swap(metadata.transform_T);
    std::vector<float>().swap(metadata.U);
}

}

Index::Index(size_t d, size_t nlist, size_t nprobe, MetricType metric, OptLevel opt_level, size_t sub_k, size_t sub_nlist, size_t sub_nprobe, bool verbose, EdgeDevice edge_device_enabled)
    : d(d), nlist(nlist), nprobe(nprobe), metric(metric), opt_level(opt_level), sub_k(sub_k), sub_nlist(sub_nlist), sub_nprobe(sub_nprobe), verbose(verbose), edge_device_enabled(edge_device_enabled) {
    lists = std::make_unique<IVF[]>(nlist);
    centroid_codes = std::make_unique<float[]>(nlist * d);
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
}

void Index::configure_multipivot(MultiPivotScope scope,
                                 const std::string& method,
                                 size_t pivot_count,
                                 uint64_t seed,
                                 size_t irls_max_iter_value,
                                 float irls_residual_floor_value,
                                 GreedyPruneConfig greedy_config) {
    if (metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("Multi-pivot bounds are L2-only");
    }
    if (!validMultipivotMethod(method)) {
        throw std::invalid_argument(
            "multipivot method must be affine_fps, pca, weighted_pca, irls_pca, or greedy_prune");
    }
    if (!validPivotCount(pivot_count)) {
        throw std::invalid_argument("pivot_count must be in 1..MULTIPIVOT_MAX_PIVOTS");
    }
    if (method == "irls_pca") {
        if (irls_max_iter_value == 0) {
            throw std::invalid_argument("irls_max_iter must be >= 1");
        }
        if (!(irls_residual_floor_value > 0.0f) || !std::isfinite(irls_residual_floor_value)) {
            throw std::invalid_argument("irls_residual_floor must be a positive finite float");
        }
    }
    if (method == "greedy_prune") {
        if (greedy_config.pseudo_nq == 0 || greedy_config.negatives_per_q == 0) {
            throw std::invalid_argument("greedy_prune requires pseudo_nq and negatives_per_q >= 1");
        }
        if (greedy_config.power_iters == 0 || greedy_config.proxy_candidates == 0) {
            throw std::invalid_argument("greedy_prune requires power_iters and proxy_candidates >= 1");
        }
    }
    multipivot_scope = scope;
    multipivot_method = method;
    multipivot_built_count = pivot_count;
    multipivot_active_count = pivot_count;
    multipivot_seed = seed;
    irls_max_iter = irls_max_iter_value;
    irls_residual_floor = irls_residual_floor_value;
    greedy_prune_config = greedy_config;
}

void Index::rebuild_multipivot_metadata() {
    if (multipivot_built_count == 0) {
        global_pivots.clear();
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            lists[list_id].pivots.clear();
        }
        multipivot_selection_seconds = 0;
        multipivot_candidate_distance_seconds = 0;
        multipivot_build_distance_computations = 0;
        return;
    }
    if (metric != MetricType::METRIC_L2) {
        throw std::runtime_error("Cannot build multi-pivot metadata for a non-L2 index");
    }
    if (!lists || !centroid_codes) {
        throw std::runtime_error("Cannot rebuild multipivot metadata without a loaded IVF");
    }
    multipivot_selection_seconds = 0;
    multipivot_candidate_distance_seconds = 0;
    multipivot_build_distance_computations = 0;
    size_t total_candidates = 0;
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        total_candidates += lists[list_id].list_size;
    }
    auto select_pivots = [&](const float* codes, const idx_t* ids, size_t n,
                             const float* forced_anchor = nullptr,
                             idx_t forced_id = -1) -> PivotSelectionResult {
        if (multipivot_method == "pca") {
            return selectPcaPivots(codes, n, d, multipivot_built_count, forced_anchor, forced_id);
        }
        if (multipivot_method == "weighted_pca") {
            // No CLI sample weights yet: uniform weights => fixed-center PCA.
            return selectWeightedPcaPivots(codes, n, d, multipivot_built_count, forced_anchor,
                                          forced_id, nullptr);
        }
        if (multipivot_method == "irls_pca") {
            return selectIrlsPcaPivots(codes, n, d, multipivot_built_count, forced_anchor,
                                      forced_id, irls_max_iter, irls_residual_floor, nullptr);
        }
        if (multipivot_method == "greedy_prune") {
            // Build-time only: directions are frozen into pivot codes; query never updates them.
            return selectGreedyPrunePivots(codes, n, d, multipivot_built_count, forced_anchor,
                                          forced_id, multipivot_seed, greedy_prune_config,
                                          /*train_queries=*/nullptr, /*train_query_count=*/0);
        }
        return selectAffineFps(codes, ids, n, d, multipivot_built_count, multipivot_seed,
                               forced_anchor, forced_id);
    };

    pca_cov_eigenvalues.clear();
    irls_objective_history.clear();
    greedy_gain_history.clear();
    if (multipivot_scope == MultiPivotScope::GLOBAL) {
        // Global PCA-family: μ = mean of IVF centroids (candidates here).
        PivotSelectionResult selected =
            select_pivots(centroid_codes.get(), centroid_ids.get(), nlist);
        global_pivots = std::move(selected.metadata);
        pca_cov_eigenvalues = std::move(selected.pca_cov_eigenvalues);
        irls_objective_history = std::move(selected.irls_objective_history);
        greedy_gain_history = std::move(selected.greedy_gain_history);
        multipivot_selection_seconds = selected.elapsed_seconds;
        const auto distance_start = std::chrono::steady_clock::now();
#pragma omp parallel for
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            lists[list_id].pivots = global_pivots;
            computeCandidatePivotDistances(lists[list_id].candidate_codes.get(),
                                           lists[list_id].list_size, d, lists[list_id].pivots);
            lists[list_id].pivots.external_projection_geometry = true;
            discard_global_pivot_geometry(lists[list_id].pivots);
        }
        multipivot_candidate_distance_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - distance_start).count();
        multipivot_build_distance_computations = total_candidates * global_pivots.count;
    } else {
        global_pivots.clear();
        const bool pca_centroid_anchor = isCentroidAnchoredMultipivotMethod(multipivot_method);
        // Per-list real-query assignment for greedy_prune (top-K nearest lists).
        std::vector<std::vector<float>> list_train_queries(nlist);
        std::vector<size_t> list_train_nq(nlist, 0);
        if (multipivot_method == "greedy_prune" && greedy_train_nq > 0 &&
            !greedy_train_queries.empty()) {
            const size_t K = std::max<size_t>(
                1, std::min(greedy_prune_config.real_lists_per_query == 0
                                ? 1
                                : greedy_prune_config.real_lists_per_query,
                            nlist));
            std::vector<std::pair<float, size_t>> dists(nlist);
            for (size_t qi = 0; qi < greedy_train_nq; ++qi) {
                const float* q = greedy_train_queries.data() + qi * d;
                for (size_t lid = 0; lid < nlist; ++lid) {
                    dists[lid] = {calculatedEuclideanDistance(q, centroid_codes.get() + lid * d, d),
                                  lid};
                }
                std::partial_sort(dists.begin(), dists.begin() + static_cast<std::ptrdiff_t>(K),
                                  dists.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                for (size_t k = 0; k < K; ++k) {
                    const size_t lid = dists[k].second;
                    list_train_queries[lid].insert(list_train_queries[lid].end(), q, q + d);
                    list_train_nq[lid] += 1;
                }
            }
            if (verbose) {
                size_t nonempty = 0, total_assign = 0;
                for (size_t lid = 0; lid < nlist; ++lid) {
                    if (list_train_nq[lid] > 0) ++nonempty;
                    total_assign += list_train_nq[lid];
                }
                std::cout << std::format(
                    "greedy real-query train: nq={} ratio={:g} lists_per_q={} "
                    "assigned_lists={}/{} total_assign={}\n",
                    greedy_train_nq, greedy_prune_config.real_query_ratio, K, nonempty, nlist,
                    total_assign);
            }
        }
        const auto selection_start = std::chrono::steady_clock::now();
        // Per-list PCA/IRLS/FPS are independent; parallelize across lists.
        // Keep Eigen single-threaded inside each task to avoid OpenMP oversubscription.
        const int eigen_threads_prev = Eigen::nbThreads();
        Eigen::setNbThreads(1);
        std::atomic<bool> diagnostics_captured{false};
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            IVF& list = lists[list_id];
            std::vector<idx_t> source_ids(list.list_size);
            std::transform(list.candidate_id.get(), list.candidate_id.get() + list.list_size,
                           source_ids.begin(), [](size_t id) { return static_cast<idx_t>(id); });
            const float* centroid = centroid_codes.get() + list_id * d;
            PivotSelectionResult selected;
            if (multipivot_method == "greedy_prune") {
                const float* tq =
                    list_train_nq[list_id] > 0 ? list_train_queries[list_id].data() : nullptr;
                const size_t tnq = list_train_nq[list_id];
                selected = selectGreedyPrunePivots(
                    list.candidate_codes.get(), list.list_size, d, multipivot_built_count,
                    pca_centroid_anchor ? centroid : nullptr,
                    pca_centroid_anchor ? static_cast<idx_t>(list_id) : static_cast<idx_t>(-1),
                    multipivot_seed + list_id, greedy_prune_config, tq, tnq);
            } else if (pca_centroid_anchor) {
                selected = select_pivots(list.candidate_codes.get(), source_ids.data(),
                                         list.list_size, centroid, static_cast<idx_t>(list_id));
            } else {
                selected = select_pivots(list.candidate_codes.get(), source_ids.data(),
                                         list.list_size);
            }
            // Keep one representative diagnostics snapshot (any finished list).
            if (!diagnostics_captured.exchange(true)) {
                pca_cov_eigenvalues = std::move(selected.pca_cov_eigenvalues);
                irls_objective_history = std::move(selected.irls_objective_history);
                greedy_gain_history = std::move(selected.greedy_gain_history);
            }
            list.pivots = std::move(selected.metadata);
        }
        Eigen::setNbThreads(eigen_threads_prev);
        multipivot_selection_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - selection_start).count();
        if (verbose) {
            std::cout << std::format(
                "multipivot per_list selection: method={} P={} lists={} elapsed={:.3f}s "
                "(omp_threads={})\n",
                multipivot_method, multipivot_built_count, nlist, multipivot_selection_seconds,
                omp_get_max_threads());
        }
        const auto distance_start = std::chrono::steady_clock::now();
        uint64_t distance_computations = 0;
#pragma omp parallel for reduction(+ : distance_computations)
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            IVF& list = lists[list_id];
            computeCandidatePivotDistances(list.candidate_codes.get(), list.list_size, d,
                                           list.pivots);
            distance_computations += list.list_size * list.pivots.count;
        }
        multipivot_build_distance_computations = distance_computations;
        multipivot_candidate_distance_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - distance_start).count();
    }
    rebuild_microblocks();
}

void Index::apply_signature_precision() {
    if (signature_precision == SignaturePrecision::FLOAT32 || !lists) return;
    if (!global_pivots.signature_data.empty()) {
        compressCandidateSignatures(global_pivots, signature_precision);
    }
#pragma omp parallel for
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        if (!lists[list_id].pivots.signature_data.empty()) {
            compressCandidateSignatures(lists[list_id].pivots, signature_precision);
        }
    }
    if (verbose) {
        std::cout << "signature_precision=" << signaturePrecisionName(signature_precision)
                  << std::endl;
    }
}

void Index::set_projection_prefix_length(size_t prefix_length) {
    projection_prefix_length = prefix_length;
    prepareCandidatePrefixResiduals(global_pivots, prefix_length);
    if (!lists) return;
#pragma omp parallel for
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        prepareCandidatePrefixResiduals(lists[list_id].pivots, prefix_length);
    }
}

void Index::rebuild_microblocks() {
    microblock_build_seconds = 0;
    if (!lists) return;
    if (!microblock_enabled || multipivot_built_count == 0) {
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            lists[list_id].microblocks.clear();
        }
        return;
    }
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        IVF& list = lists[list_id];
        const size_t built_P =
            list.pivots.count == multipivot_built_count ? multipivot_built_count : 0;
        if (built_P == 0 || !list.pivots.usable_signature(built_P)) {
            list.microblocks.clear();
            continue;
        }
        list.microblocks = buildListMicroblocks(list.pivots, d, built_P, microblock_bytes,
                                                multipivot_seed + list_id);
    }
    microblock_build_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (verbose) {
        size_t total_blocks = 0;
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            total_blocks += lists[list_id].microblocks.size();
        }
        std::cout << "microblocks built: lists=" << nlist << " blocks=" << total_blocks
                  << " bytes=" << microblock_bytes << " elapsed=" << microblock_build_seconds
                  << "s" << std::endl;
    }
}

void Index::set_greedy_train_queries(const float* queries, size_t n) {
    if (queries == nullptr || n == 0 || d == 0) {
        greedy_train_queries.clear();
        greedy_train_nq = 0;
        return;
    }
    greedy_train_nq = n;
    greedy_train_queries.resize(n * d);
    std::memcpy(greedy_train_queries.data(), queries, sizeof(float) * n * d);
}

void Index::set_multipivot_search(MultiPivotMode mode, size_t pivot_count) {
    if (mode != MultiPivotMode::NONE && metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("Multi-pivot bounds are L2-only");
    }
    if (mode != MultiPivotMode::NONE && !validPivotCount(pivot_count)) {
        throw std::invalid_argument("pivot_count must be in 1..MULTIPIVOT_MAX_PIVOTS");
    }
    if (mode != MultiPivotMode::NONE && pivot_count > multipivot_built_count) {
        throw std::invalid_argument(
            "Requested pivot prefix exceeds the persisted signature count");
    }
    multipivot_mode = mode;
    multipivot_active_count = mode == MultiPivotMode::NONE ? 0 : pivot_count;
}

Index& Index::operator=(Index&& other) noexcept {
    d = other.d;
    nlist = other.nlist;
    nprobe = other.nprobe;
    metric = other.metric;
    opt_level = other.opt_level;
    added_opt_level = other.added_opt_level;
    sub_k = other.sub_k;
    sub_nlist = other.sub_nlist;
    sub_nprobe = other.sub_nprobe;
    verbose = other.verbose;
    edge_device_enabled = other.edge_device_enabled;
    collect_list_stats = other.collect_list_stats;
    multipivot_scope = other.multipivot_scope;
    multipivot_mode = other.multipivot_mode;
    multipivot_method = std::move(other.multipivot_method);
    multipivot_built_count = other.multipivot_built_count;
    multipivot_active_count = other.multipivot_active_count;
    multipivot_seed = other.multipivot_seed;
    irls_max_iter = other.irls_max_iter;
    irls_residual_floor = other.irls_residual_floor;
    greedy_prune_config = other.greedy_prune_config;
    greedy_train_queries = std::move(other.greedy_train_queries);
    greedy_train_nq = other.greedy_train_nq;
    other.greedy_train_nq = 0;
    global_pivots = std::move(other.global_pivots);
    multipivot_selection_seconds = other.multipivot_selection_seconds;
    multipivot_candidate_distance_seconds = other.multipivot_candidate_distance_seconds;
    multipivot_build_distance_computations = other.multipivot_build_distance_computations;
    pca_cov_eigenvalues = std::move(other.pca_cov_eigenvalues);
    irls_objective_history = std::move(other.irls_objective_history);
    greedy_gain_history = std::move(other.greedy_gain_history);
    signature_precision = other.signature_precision;
    microblock_enabled = other.microblock_enabled;
    microblock_bytes = other.microblock_bytes;
    microblock_build_seconds = other.microblock_build_seconds;
    lists = std::move(other.lists);
    centroid_codes = std::move(other.centroid_codes);
    centroid_ids = std::move(other.centroid_ids);
    return *this;
}

void Index::train(size_t n, const float* codes, bool faiss, bool lite) {
    // 这里假设Clustering类已经定义好，并且有一个合适的构造函数和train方法
    auto tic1 = std::chrono::high_resolution_clock::now();
    if (!faiss) {
        ClusteringParameters cp;
        cp.metric = this->metric;
        cp.niter = 20;                     // 或其他合适的值
        if (lite) {
            cp.niter = 2;
        }
        cp.seed = 6666;                    // 或其他合适的值
        cp.max_points_per_centroid = 256;  // 或其他合适的值

        Clustering clustering(this->d, this->nlist, verbose, cp);
        clustering.train(n, codes);

        this->centroid_codes.reset(clustering.get_centroids());
    } else {
        faiss::IndexFlatL2 quantizer(d);  // the other index
        faiss::IndexIVFFlat index(&quantizer, d, nlist);
        index.train(n, codes);
        this->centroid_codes = std::make_unique<float[]>(nlist * d);
        std::copy_n(quantizer.get_xb(), nlist * d, this->centroid_codes.get());
    }
    if (metric == MetricType::METRIC_IP) {
        float* codes = this->centroid_codes.get();
#pragma omp parallel for
        for (size_t i = 0; i < nlist; i++) {
            float norm = calculatedInnerProduct(codes + i * d, codes + i * d, d);
            if (norm != 0) {
                for (size_t j = 0; j < d; j++) {
                    codes[i * d + j] /= sqrt(norm);
                }
            }
        }
    }
    auto tic2 = std::chrono::high_resolution_clock::now();
    if (verbose) {
        std::cout << std::format("train elapsed: {}s\n", std::chrono::duration<double>(tic2 - tic1).count());
    }
}

void Index::train_ivf_hnsw(size_t n,
                           const float* codes,
                           size_t hnsw_m,
                           size_t ef_construction,
                           size_t ef_search,
                           size_t niter,
                           int seed) {
    if (metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("IVF+HNSW coarse clustering currently requires L2");
    }
    if (n < nlist || hnsw_m == 0 || ef_construction == 0 || ef_search == 0 ||
        niter == 0) {
        throw std::invalid_argument("Invalid IVF+HNSW coarse clustering parameters");
    }
    faiss::ClusteringParameters parameters;
    parameters.niter = static_cast<int>(niter);
    parameters.nredo = 1;
    parameters.seed = seed;
    parameters.max_points_per_centroid = 256;
    parameters.min_points_per_centroid = 1;
    parameters.verbose = true;

    faiss::Clustering clustering(static_cast<int>(d), static_cast<int>(nlist), parameters);
    faiss::IndexHNSWFlat assignment(
        static_cast<int>(d), static_cast<int>(hnsw_m), faiss::METRIC_L2);
    assignment.hnsw.efConstruction = static_cast<int>(ef_construction);
    assignment.hnsw.efSearch = static_cast<int>(ef_search);
    assignment.verbose = verbose;

    std::cout << std::format(
        "IVF+HNSW clustering: n={} d={} nlist={} iterations={} M={} efConstruction={} "
        "efSearch={} seed={}\n",
        n, d, nlist, niter, hnsw_m, ef_construction, ef_search, seed)
              << std::flush;
    clustering.train(static_cast<faiss::idx_t>(n), codes, assignment);
    if (clustering.centroids.size() != nlist * d) {
        throw std::runtime_error("IVF+HNSW clustering returned an invalid centroid table");
    }
    centroid_codes = std::make_unique<float[]>(nlist * d);
    std::copy(clustering.centroids.begin(), clustering.centroids.end(), centroid_codes.get());
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
    std::cout << "IVF+HNSW clustering completed" << std::endl;
}

namespace {
constexpr char CENTROID_MAGIC[8] = {'T', 'B', 'C', 'E', 'N', 'T', 'R', '1'};
constexpr uint32_t CENTROID_FORMAT_VERSION = 1;
}  // namespace

void Index::save_centroids(const std::string& path) const {
    if (!centroid_codes || nlist == 0 || d == 0) {
        throw std::runtime_error("Cannot save centroids: index has no trained centroids");
    }
    prepareDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open centroids file for write: " + path);
    }
    out.write(CENTROID_MAGIC, sizeof(CENTROID_MAGIC));
    const uint32_t version = CENTROID_FORMAT_VERSION;
    const uint32_t endian = INDEX_ENDIAN_MARKER;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&endian), sizeof(endian));
    out.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(MetricType));
    out.write(reinterpret_cast<const char*>(centroid_codes.get()),
              static_cast<std::streamsize>(nlist * d * sizeof(float)));
    if (!out) throw std::runtime_error("Failed while writing centroids: " + path);
    if (verbose) {
        std::cout << std::format("centroids saved: {} (nlist={} d={})\n", path, nlist, d);
    }
}

void Index::load_centroids(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open centroids file: " + path);
    }
    char magic[sizeof(CENTROID_MAGIC)]{};
    in.read(magic, sizeof(magic));
    uint32_t version = 0;
    uint32_t endian = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&endian), sizeof(endian));
    if (!in || std::memcmp(magic, CENTROID_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid centroids file magic: " + path);
    }
    if (version != CENTROID_FORMAT_VERSION || endian != INDEX_ENDIAN_MARKER) {
        throw std::runtime_error("Unsupported centroids format/version: " + path);
    }
    size_t file_d = 0;
    size_t file_nlist = 0;
    MetricType file_metric = MetricType::METRIC_L2;
    in.read(reinterpret_cast<char*>(&file_d), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&file_nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&file_metric), sizeof(MetricType));
    if (!in || file_d == 0 || file_nlist == 0) {
        throw std::runtime_error("Corrupt centroids header: " + path);
    }
    if (d != 0 && file_d != d) {
        throw std::runtime_error("centroids d mismatch");
    }
    if (nlist != 0 && file_nlist != nlist) {
        throw std::runtime_error("centroids nlist mismatch");
    }
    if (file_metric != metric) {
        throw std::runtime_error("centroids metric mismatch");
    }
    d = file_d;
    nlist = file_nlist;
    centroid_codes = std::make_unique<float[]>(nlist * d);
    in.read(reinterpret_cast<char*>(centroid_codes.get()),
            static_cast<std::streamsize>(nlist * d * sizeof(float)));
    if (!in) throw std::runtime_error("Failed while reading centroids: " + path);
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
    if (!lists) {
        lists = std::make_unique<IVF[]>(nlist);
    }
    if (verbose) {
        std::cout << std::format("centroids loaded: {} (nlist={} d={})\n", path, nlist, d);
    }
}

void Index::ensure_triangle_radii() {
    if (metric != MetricType::METRIC_L2) {
        throw std::runtime_error("ensure_triangle_radii is L2-only");
    }
    if (!lists || !centroid_codes) {
        throw std::runtime_error("ensure_triangle_radii requires loaded IVF + centroids");
    }
#pragma omp parallel for
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        IVF& list = lists[list_id];
        if (list.list_size == 0) continue;
        list.ensure_opt_storage(OptLevel::OPT_TRIANGLE, sub_k);
        const float* centroid = centroid_codes.get() + list_id * d;
        for (size_t i = 0; i < list.list_size; ++i) {
            const float dis =
                calculatedEuclideanDistance(list.get_candidate_codes(i), centroid, d);
            list.candidate2centroid[i] = dis;
            list.sqrt_candidate2centroid[i] = std::sqrt(dis);
        }
    }
    added_opt_level = static_cast<OptLevel>(static_cast<int>(added_opt_level) |
                                            static_cast<int>(OptLevel::OPT_TRIANGLE));
    if (verbose) {
        std::cout << "triangle radii materialized from shared clustering\n";
    }
}

void Index::upgrade_pruning(OptLevel target) {
    if (!lists || !centroid_codes) {
        throw std::runtime_error("upgrade_pruning requires a loaded IVF with centroids");
    }
    const int need = static_cast<int>(target);
    if (need & (static_cast<int>(OptLevel::OPT_TRIANGLE) | static_cast<int>(OptLevel::OPT_SUBNN_IP))) {
        ensure_triangle_radii();
    }
    // Temporarily expose full target bits so build_subnn_structures sees SUBNN flags.
    const OptLevel prev_opt = opt_level;
    opt_level = static_cast<OptLevel>(static_cast<int>(opt_level) | need);
    if (need & (static_cast<int>(OptLevel::OPT_SUBNN_L2) | static_cast<int>(OptLevel::OPT_SUBNN_IP))) {
#pragma omp parallel for
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            if (lists[list_id].list_size == 0) continue;
            lists[list_id].ensure_opt_storage(target, sub_k);
        }
        build_subnn_structures();
    }
    added_opt_level = static_cast<OptLevel>(static_cast<int>(added_opt_level) | need);
    opt_level = prev_opt;
    if (multipivot_built_count > 0) {
        rebuild_multipivot_metadata();
    } else {
        global_pivots.clear();
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            lists[list_id].pivots.clear();
            lists[list_id].microblocks.clear();
        }
    }
    if (verbose) {
        std::cout << std::format("upgrade_pruning done: added_opt={} mp_P={}\n",
                                 static_cast<int>(added_opt_level), multipivot_built_count);
    }
}

std::unique_ptr<IVFScanBase> Index::get_scanner(MetricType metric, OptLevel opt_level, size_t k, EdgeDevice edge_device_enabled) {
    if (metric == MetricType::METRIC_L2) {
        if (edge_device_enabled) {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_TRI_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_TRI_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_SUBNN_ONLY:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_ONLY, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_ALL:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_ALL, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        } else {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_TRI_SUBNN_L2:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_TRI_SUBNN_IP:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_SUBNN_ONLY:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_ONLY, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_ALL:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_ALL, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        }
    } else {
        if (edge_device_enabled) {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_SUBNN_ONLY:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_ONLY, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                // case OptLevel::OPT_ALL:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_ALL, EdgeDevice::EDGEDEVIVE_ENABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        } else {
            switch (opt_level) {
                case OptLevel::OPT_NONE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                case OptLevel::OPT_TRIANGLE:
                    return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRIANGLE, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_L2:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_L2, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_TRI_SUBNN_IP:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_IP, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_SUBNN_ONLY:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_ONLY, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                // case OptLevel::OPT_ALL:
                //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_ALL, EdgeDevice::EDGEDEVIVE_DISABLED>(d, k));
                default:
                    throw std::runtime_error("Unsupported opt_level");
            }
        }
    }
};

void Index::single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels) {
    if (n == 0) {
        return;
    }
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, 1);
    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist,
                                          centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          distances + i,
                                          labels + i);
        // no need to sort result, because only one result
    }
}

void Index::add(size_t n, const float* codes) {
    auto tic1 = std::chrono::high_resolution_clock::now();
    if (n == 0) {
        auto tic2 = std::chrono::high_resolution_clock::now();
        if (verbose) {
            std::cout << std::format("add elapsed: {}s\n", std::chrono::duration<double>(tic2 - tic1).count());
        }
        return;
    }

    added_opt_level = opt_level;
    std::unique_ptr<float[]> candicate2centroid = std::make_unique<float[]>(n);
    std::unique_ptr<idx_t[]> listidcandicates = std::make_unique<idx_t[]>(n);
    init_result(metric, n, candicate2centroid.get(), listidcandicates.get());
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;
    size_t extra = n % nt;
    auto nearest_search_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            single_thread_nearest_cluster_search(end - start, codes + start * d, candicate2centroid.get() + start, listidcandicates.get() + start);
        }
    }
    auto nearest_search_end = std::chrono::high_resolution_clock::now();

    if(verbose){
        std::cout << "nearest search elapsed: " << std::chrono::duration<double>(nearest_search_end - nearest_search_start).count() << "s" << std::endl;
    }

    auto sort_and_add_start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<size_t[]> list_sizes = std::make_unique<size_t[]>(nlist);
    std::fill_n(list_sizes.get(), nlist, 0);

    // TODO: parallelize this part ?
    for (size_t i = 0; i < n; i++) {
        list_sizes[listidcandicates[i]]++;
    }

    size_t total_add = 0;
    for (size_t i = 0; i < nlist; i++) {
        total_add += list_sizes[i];
    }

#pragma omp parallel for
    for (size_t i = 0; i < nlist; i++) {
        lists[i].reset(list_sizes[i], d, sub_k, added_opt_level);
    }

    std::fill_n(list_sizes.get(), nlist, 0);

    std::unique_ptr<size_t[]> add_order = std::make_unique<size_t[]>(n);
    std::iota(add_order.get(), add_order.get() + n, 0);
    if (metric == MetricType::METRIC_L2) {
        std::sort(add_order.get(), add_order.get() + n, [&](size_t i, size_t j) { return candicate2centroid[i] < candicate2centroid[j]; });
    } else {
        std::sort(add_order.get(), add_order.get() + n, [&](size_t i, size_t j) { return candicate2centroid[i] > candicate2centroid[j]; });
    }

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for (size_t oi = 0; oi < n; oi++) {
            size_t i = add_order[oi];
            size_t list_id = listidcandicates[i];  // assert > 0
            if (list_id % nt == tid) {
                size_t list_size = list_sizes[list_id];
                lists[list_id].candidate_id[list_size] = i;
                if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
                    lists[list_id].candidate2centroid[list_size] = candicate2centroid[i];
                }
                std::copy_n(codes + i * d, d, lists[list_id].candidate_codes.get() + list_size * d);
                list_sizes[list_id]++;
            }
        }
    }
    auto sort_and_add_end = std::chrono::high_resolution_clock::now();
    auto sort_and_add_elapsed = std::chrono::duration<double>(sort_and_add_end - sort_and_add_start).count();
    if (verbose) {
        std::cout << "sort and add elapsed: " << sort_and_add_elapsed << "s" << std::endl;
    }

    if (metric == MetricType::METRIC_L2) {
        if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
#pragma omp parallel for
            for (size_t list_id = 0; list_id < nlist; list_id++) {
                size_t list_size = lists[list_id].list_size;
                for (size_t i = 0; i < list_size; i++) {
                    lists[list_id].sqrt_candidate2centroid[i] = std::sqrt(lists[list_id].candidate2centroid[i]);
                }
            }
        }

#pragma omp parallel for
        for (size_t list_id = 0; list_id < nlist; list_id++) {
            size_t list_size = lists[list_id].list_size;
            for (size_t i = 0; i < list_size; i++) {
                const float* code = codes + lists[list_id].candidate_id[i] * d;
                lists[list_id].candidate_norms[i] = calculatedInnerProduct(code, code, d);
            }
        }
    }

    rebuild_multipivot_metadata();
    build_subnn_structures();

    auto tic2 = std::chrono::high_resolution_clock::now();
    if (verbose) {
        std::cout << std::format("add elapsed: {}s\n", std::chrono::duration<double>(tic2 - tic1).count());
    }
}

void Index::add_batched(size_t n,
                        size_t batch_size,
                        const BatchLoader& loader,
                        HnswAssignmentConfig hnsw_config) {
    const auto started = std::chrono::steady_clock::now();
    if (n == 0) return;
    if (batch_size == 0 || !loader) {
        throw std::invalid_argument("add_batched requires a non-zero batch size and loader");
    }
    if (!centroid_codes || !lists) {
        throw std::runtime_error("add_batched requires trained centroids");
    }

    added_opt_level = opt_level;
    std::vector<size_t> list_counts(nlist, 0);
    if (nlist > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("add_batched supports at most uint32_t list IDs");
    }
    const size_t progress_step = std::max<size_t>(1, (n + 19) / 20);
    size_t next_count_progress = progress_step;
    size_t next_fill_progress = progress_step;

    std::unique_ptr<faiss::IndexHNSWFlat> hnsw_assignment;
    if (hnsw_config.enabled) {
        if (metric != MetricType::METRIC_L2 || hnsw_config.m == 0 ||
            hnsw_config.ef_construction == 0 || hnsw_config.ef_search == 0) {
            throw std::invalid_argument("Invalid HNSW batch-assignment configuration");
        }
        hnsw_assignment = std::make_unique<faiss::IndexHNSWFlat>(
            static_cast<int>(d), static_cast<int>(hnsw_config.m), faiss::METRIC_L2);
        hnsw_assignment->hnsw.efConstruction =
            static_cast<int>(hnsw_config.ef_construction);
        hnsw_assignment->hnsw.efSearch = static_cast<int>(hnsw_config.ef_search);
        hnsw_assignment->add(static_cast<faiss::idx_t>(nlist), centroid_codes.get());
        std::cout << std::format(
            "Streaming byte index: HNSW assignment ready M={} efConstruction={} "
            "efSearch={} centroids={}\n",
            hnsw_config.m, hnsw_config.ef_construction, hnsw_config.ef_search, nlist)
                  << std::flush;
    }

    auto assign_batch = [&](const float* codes, size_t rows,
                            float* distances, idx_t* labels) {
        init_result(metric, rows, distances, labels);
        if (hnsw_assignment) {
            // Do not hand the full decoded build batch to Faiss at once.
            // A 262144-vector I/O batch is desirable for sequential bvecs
            // reads, but it needlessly makes HNSW's temporary result/scratch
            // allocation large. Splitting affects neither the graph nor the
            // selected nearest centroid.
            const size_t query_batch = std::max<size_t>(
                1, std::min(hnsw_config.query_batch_size, rows));
            for (size_t offset = 0; offset < rows; offset += query_batch) {
                const size_t count = std::min(query_batch, rows - offset);
                hnsw_assignment->search(
                    static_cast<faiss::idx_t>(count), codes + offset * d, 1,
                    distances + offset,
                    reinterpret_cast<faiss::idx_t*>(labels + offset));
            }
            return;
        }
        const size_t threads =
            std::min(static_cast<size_t>(omp_get_max_threads()), rows);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t worker = 0; worker < threads; ++worker) {
            const size_t worker_begin = rows * worker / threads;
            const size_t worker_end = rows * (worker + 1) / threads;
            single_thread_nearest_cluster_search(
                worker_end - worker_begin, codes + worker_begin * d,
                distances + worker_begin, labels + worker_begin);
        }
    };

    // Pass 1 only keeps one decoded batch and determines exact list sizes.
    for (size_t begin = 0; begin < n; begin += batch_size) {
        const size_t rows = std::min(batch_size, n - begin);
        auto codes = loader(begin, rows);
        if (!codes) throw std::runtime_error("add_batched loader returned null");
        std::vector<float> distances(rows);
        std::vector<idx_t> labels(rows);
        assign_batch(codes.get(), rows, distances.data(), labels.data());
        for (size_t row = 0; row < rows; ++row) {
            const idx_t label = labels[row];
            if (label < 0 || static_cast<size_t>(label) >= nlist) {
                throw std::runtime_error("add_batched received an invalid list assignment");
            }
            ++list_counts[static_cast<size_t>(label)];
        }
        if (begin + rows >= next_count_progress || begin + rows == n) {
            std::cout << std::format("Streaming byte index: assignment {:.1f}%\n",
                                     100.0 * (begin + rows) / n);
            while (next_count_progress <= begin + rows && next_count_progress < n) {
                next_count_progress += progress_step;
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        lists[list_id].reset(list_counts[list_id], d, sub_k, added_opt_level);
    }

    auto write_offsets = std::make_unique<std::atomic<size_t>[]>(nlist);
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        write_offsets[list_id].store(0, std::memory_order_relaxed);
    }

    // Pass 2 repeats coarse assignment and fills the allocated lists.  Keeping
    // one uint32 assignment per database vector costs 3.73 GiB at SIFT1B;
    // re-running the bounded HNSW lookup is cheaper than reserving that global
    // array and gives the same deterministic list ID. candidate_norms
    // temporarily stores centroid distance and is replaced with vector norm
    // after sorting.
    for (size_t begin = 0; begin < n; begin += batch_size) {
        const size_t rows = std::min(batch_size, n - begin);
        auto codes = loader(begin, rows);
        if (!codes) throw std::runtime_error("add_batched loader returned null");
        std::vector<float> distances(rows);
        std::vector<idx_t> labels(rows);
        assign_batch(codes.get(), rows, distances.data(), labels.data());
        for (size_t row = 0; row < rows; ++row) {
            if (labels[row] < 0 || static_cast<size_t>(labels[row]) >= nlist) {
                throw std::runtime_error("add_batched received an invalid list assignment");
            }
        }
#pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row) {
            const size_t list_id = static_cast<size_t>(labels[row]);
            const size_t offset =
                write_offsets[list_id].fetch_add(1, std::memory_order_relaxed);
            IVF& list = lists[list_id];
            list.candidate_id[offset] = begin + row;
            list.candidate_norms[offset] = distances[row];
            std::copy_n(codes.get() + row * d, d,
                        list.candidate_codes.get() + offset * d);
        }
        if (begin + rows >= next_fill_progress || begin + rows == n) {
            std::cout << std::format("Streaming byte index: fill {:.1f}%\n",
                                     100.0 * (begin + rows) / n);
            while (next_fill_progress <= begin + rows && next_fill_progress < n) {
                next_fill_progress += progress_step;
            }
        }
    }

    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        if (write_offsets[list_id].load(std::memory_order_relaxed) !=
            list_counts[list_id]) {
            throw std::runtime_error("add_batched list size changed between passes");
        }
    }

    // Triangle filtering assumes ascending centroid radius for L2 (descending
    // similarity for IP). Reorder each list in place: allocating sorted_codes
    // duplicated up to the full 476.8 GiB code store at SIFT1B.
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t list_id = 0; list_id < nlist; ++list_id) {
        IVF& list = lists[list_id];
        const size_t count = list.list_size;
        if (count == 0) continue;
        std::vector<size_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        const float* assignment_distance = list.candidate_norms.get();
        std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
            if (assignment_distance[lhs] == assignment_distance[rhs]) {
                return list.candidate_id[lhs] < list.candidate_id[rhs];
            }
            return metric == MetricType::METRIC_L2
                ? assignment_distance[lhs] < assignment_distance[rhs]
                : assignment_distance[lhs] > assignment_distance[rhs];
        });

        // order is destination -> source. Invert it in place into
        // source -> destination, using its high bit as a visited marker.
        constexpr size_t visited = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
        if (count >= visited) {
            throw std::runtime_error("IVF list too large for in-place permutation");
        }
        for (size_t start = 0; start < count; ++start) {
            if (order[start] & visited) continue;
            size_t current = start;
            size_t next = order[current];
            while (next != start) {
                const size_t next_next = order[next];
                order[next] = current | visited;
                current = next;
                next = next_next;
            }
            order[start] = current | visited;
        }

        std::vector<float> code_tmp(d);
        for (size_t start = 0; start < count; ++start) {
            if ((order[start] & visited) == 0) continue;
            size_t current = start;
            std::copy_n(list.candidate_codes.get() + start * d, d, code_tmp.data());
            size_t id_tmp = list.candidate_id[start];
            float distance_tmp = list.candidate_norms[start];
            while (true) {
                const size_t destination = order[current] & ~visited;
                order[current] = destination;
                if (destination == start) {
                    std::copy_n(code_tmp.data(), d,
                                list.candidate_codes.get() + destination * d);
                    list.candidate_id[destination] = id_tmp;
                    list.candidate_norms[destination] = distance_tmp;
                    break;
                }
                float* destination_code = list.candidate_codes.get() + destination * d;
                for (size_t axis = 0; axis < d; ++axis) {
                    std::swap(code_tmp[axis], destination_code[axis]);
                }
                std::swap(id_tmp, list.candidate_id[destination]);
                std::swap(distance_tmp, list.candidate_norms[destination]);
                current = destination;
            }
        }
        if (list.candidate2centroid) {
            for (size_t row = 0; row < count; ++row) {
                list.candidate2centroid[row] = list.candidate_norms[row];
                list.sqrt_candidate2centroid[row] =
                    std::sqrt(std::max(0.0f, list.candidate_norms[row]));
            }
        }
        if (metric == MetricType::METRIC_L2) {
            for (size_t row = 0; row < count; ++row) {
                const float* code = list.candidate_codes.get() + row * d;
                list.candidate_norms[row] = calculatedInnerProduct(code, code, d);
            }
        }
    }

    std::cout << "Streaming byte index: list sorting completed; building pruning metadata"
              << std::endl;

    rebuild_multipivot_metadata();
    build_subnn_structures();
    std::cout << "Streaming byte index: IVF build completed" << std::endl;
    if (verbose) {
        std::cout << std::format(
            "streaming add elapsed: {}s\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    }
}

void Index::build_subnn_structures() {
    if (metric != MetricType::METRIC_L2) {
        return;
    }
    const bool need_l2 = (opt_level & OptLevel::OPT_SUBNN_L2) != 0;
    const bool need_ip = (opt_level & OptLevel::OPT_SUBNN_IP) != 0;
    if (!need_l2 && !need_ip) {
        return;
    }
    if (!lists || !centroid_codes) {
        throw std::runtime_error("build_subnn_structures requires loaded IVF + centroids");
    }

    size_t total_processd = 0;
    size_t total_sub_count_ip = 0;
    size_t total_sub_recall_ip = 0;
    size_t total_sub_count_l2 = 0;
    size_t total_sub_recall_l2 = 0;
    size_t total_sub_count_ip_5 = 0;
    size_t total_sub_recall_ip_5 = 0;
    size_t total_sub_count_l2_5 = 0;
    size_t total_sub_recall_l2_5 = 0;
    Stopwatch logwatch;
    double train_elapsed = 0;
    double add_elapsed = 0;
    double search_elapsed = 0;
    double log_interval = 2;
    [[maybe_unused]] auto running_log = [&]() -> void {
        if (verbose) {
            if (logwatch.elapsedSeconds() > log_interval || total_processd == nlist) {
                logwatch.reset();
                std::cout << std::format("build: {:.2f}%", 100.0 * total_processd / nlist) << std::endl;
                double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
                double train_percent = total_elapsed > 0 ? 100.0 * train_elapsed / total_elapsed : 0;
                double add_percent = total_elapsed > 0 ? 100.0 * add_elapsed / total_elapsed : 0;
                double search_percent = total_elapsed > 0 ? 100.0 * search_elapsed / total_elapsed : 0;
                std::cout << std::format("train: {:.2f}%    add: {:.2f}%    search: {:.2f}%    total: {:.2f}\n",
                                         train_percent, add_percent, search_percent, total_elapsed);
                float sub_recall_ip = total_sub_count_ip ? 100.0 * total_sub_recall_ip / total_sub_count_ip : 0;
                float sub_recall_l2 = total_sub_count_l2 ? 100.0 * total_sub_recall_l2 / total_sub_count_l2 : 0;
                float sub_recall_ip_5 =
                    total_sub_count_ip_5 ? 100.0 * total_sub_recall_ip_5 / total_sub_count_ip_5 : 0;
                float sub_recall_l2_5 =
                    total_sub_count_l2_5 ? 100.0 * total_sub_recall_l2_5 / total_sub_count_l2_5 : 0;
                std::cout << std::format(
                    "Recall    SUBNN_IP top5: {:.2f}%    topk: {:.2f}%    SUBNN_L2 top5: {:.2f}%    "
                    "topk: {:.2f}%    {}/{}\n",
                    sub_recall_ip_5, sub_recall_ip, sub_recall_l2_5, sub_recall_l2, sub_nprobe, sub_nlist);
            }
        }
    };

    [[maybe_unused]] auto end_log = [&]() -> void {
        if (verbose) {
            double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
            std::cout << std::format("build: 100.0%\n");
            std::cout << std::format("train: {:.2f}   add: {:.2f}    search: {:.2f}    total: {:.2f}\n",
                                     train_elapsed, add_elapsed, search_elapsed, total_elapsed);
        }
    };
#pragma omp parallel for
    for (size_t listid = 0; listid < nlist; listid++) {
        IVF& list = lists[listid];
        const float* xb = list.get_candidate_codes();
        size_t nb = list.get_list_size();
        if (nb == 0) {
#pragma omp critical
            {
                total_processd++;
                running_log();
            }
            continue;
        }
        list.ensure_opt_storage(opt_level, sub_k);

        size_t this_sub_nlist_L2 = std::min(sub_nlist, (nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE);
        size_t this_sub_nprobe_L2 = std::min(
            std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_L2 * sub_nprobe / sub_nlist)),
            this_sub_nlist_L2);

        size_t this_sub_nlist_IP =
            std::min(std::max(1ul, static_cast<size_t>(sub_nlist)),
                     static_cast<size_t>((nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE));
        size_t this_sub_nprobe_IP = std::min(
            std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_IP * sub_nprobe / sub_nlist * IP_SUB_RATIO)),
            this_sub_nlist_IP);

        const float* centroid_code = centroid_codes.get() + listid * d;

        if (need_l2) {
            Index sub_index(d, this_sub_nlist_L2, this_sub_nprobe_L2, MetricType::METRIC_L2,
                            OptLevel::OPT_NONE, 0, 0, 0, false);
            Stopwatch watch;
            sub_index.train(nb, xb, false, true);
#pragma omp atomic
            train_elapsed += watch.elapsedSeconds(true);
            sub_index.add(nb, xb);
#pragma omp atomic
            add_elapsed += watch.elapsedSeconds(true);
            sub_index.search(nb, xb, sub_k, list.sub_nearest_L2_dis.get(), list.sub_nearest_L2_id.get());
#pragma omp atomic
            search_elapsed += watch.elapsedSeconds(true);

#ifdef SUB_STATS
            if (verbose) {
                size_t recall_nb =
                    static_cast<size_t>(1.0 * nb * RECALL_TEST_RATIO / sub_nlist * sub_nprobe);
                std::unique_ptr<float[]> recall_dis = std::make_unique<float[]>(recall_nb * sub_k);
                std::unique_ptr<idx_t[]> recall_id = std::make_unique<idx_t[]>(recall_nb * sub_k);
                sub_index.nprobe = sub_index.nlist;
                sub_index.search(recall_nb, xb, sub_k, recall_dis.get(), recall_id.get());

                for (size_t j = 0; j < recall_nb; j++) {
                    float top_recall_dis = recall_dis[j * sub_k + sub_k - 1];
                    float top_recall_dis_5 = recall_dis[j * sub_k + std::min(4ul, sub_k - 1)];
                    for (size_t k = 0; k < sub_k; k++) {
                        float dis = list.get_sub_nearest_L2_dis(j, k);
                        if (dis <= top_recall_dis) {
                            total_sub_recall_l2++;
                            if (dis <= top_recall_dis_5 && k < 5) {
                                total_sub_recall_l2_5++;
                            }
                        }
                    }
                }
                total_sub_count_l2 += recall_nb * sub_k;
                total_sub_count_l2_5 += recall_nb * std::min(5ul, sub_k);
            }
#endif

            for (size_t j = 0; j < nb * sub_k; j++) {
                list.sub_nearest_L2_dis[j] = sqrt(list.sub_nearest_L2_dis[j]);
            }
        }

        if (need_ip) {
            std::unique_ptr<float[]> norm_xb_u = std::make_unique<float[]>(nb * d);
            float* norm_xb = norm_xb_u.get();
            for (size_t j = 0; j < nb; j++) {
                float norm_xb_value = 0;
                const float* x = xb + j * d;
                for (size_t k = 0; k < d; k++) {
                    norm_xb[j * d + k] = (x[k] - centroid_code[k]);
                    norm_xb_value += norm_xb[j * d + k] * norm_xb[j * d + k];
                }

                norm_xb_value = sqrt(norm_xb_value);
                if (norm_xb_value > 0) {
                    for (size_t k = 0; k < d; k++) {
                        norm_xb[j * d + k] /= norm_xb_value;
                    }
                }
            }
            Index sub_index(d, this_sub_nlist_IP, this_sub_nprobe_IP, MetricType::METRIC_IP,
                            OptLevel::OPT_NONE, 0, 0, 0, false);
            Stopwatch watch;
            sub_index.train(nb, norm_xb);
#pragma omp atomic
            train_elapsed += watch.elapsedSeconds(true);
            sub_index.add(nb, norm_xb);
#pragma omp atomic
            add_elapsed += watch.elapsedSeconds(true);
            sub_index.search(nb, norm_xb, sub_k, list.sub_nearest_IP_dis.get(),
                             list.sub_nearest_IP_id.get());
#pragma omp atomic
            search_elapsed += watch.elapsedSeconds(true);

            for (size_t j = 0; j < nb * d; j++) {
                norm_xb[j] = -norm_xb[j];
            }

            watch.reset();
            sub_index.search(nb, norm_xb, sub_k, list.sub_farest_IP_dis.get(),
                             list.sub_farest_IP_id.get());
            search_elapsed += watch.elapsedSeconds(true);

            for (size_t j = 0; j < nb * sub_k; j++) {
                list.sub_farest_IP_dis[j] = -list.sub_farest_IP_dis[j];
            }

#ifdef SUB_STATS
            if (verbose) {
                size_t recall_nb =
                    static_cast<size_t>(1.0 * nb * RECALL_TEST_RATIO / sub_nlist * sub_nprobe);
                std::unique_ptr<float[]> recall_dis = std::make_unique<float[]>(recall_nb * sub_k);
                std::unique_ptr<idx_t[]> recall_id = std::make_unique<idx_t[]>(recall_nb * sub_k);
                sub_index.nprobe = sub_index.nlist;
                sub_index.search(recall_nb, norm_xb, sub_k, recall_dis.get(), recall_id.get());

                for (size_t j = 0; j < recall_nb; j++) {
                    float top_recall_dis = recall_dis[j * sub_k + sub_k - 1];
                    float top_recall_dis_5 = recall_dis[j * sub_k + std::min(4ul, sub_k - 1)];
                    for (size_t k = 0; k < sub_k; k++) {
                        float dis = list.get_sub_nearest_IP_dis(j, k);
                        if (dis >= top_recall_dis) {
                            total_sub_recall_ip++;
                            if (dis >= top_recall_dis_5 && k < 5) {
                                total_sub_recall_ip_5++;
                            }
                        }
                    }
                }
                total_sub_count_ip += recall_nb * sub_k;
                total_sub_count_ip_5 += recall_nb * std::min(5ul, sub_k);
            }
#endif
        }

#pragma omp critical
        {
            total_processd++;
            running_log();
        }
    }
    end_log();
    added_opt_level = static_cast<OptLevel>(
        static_cast<int>(added_opt_level) |
        (need_l2 ? static_cast<int>(OptLevel::OPT_SUBNN_L2) : 0) |
        (need_ip ? static_cast<int>(OptLevel::OPT_SUBNN_IP) : 0));
}

void Index::single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio, Stats* stats) {
#ifdef TRIBASE_ENABLE_STATS
    const auto worker_search_start = std::chrono::steady_clock::now();
#endif
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe, edge_device_enabled);
    std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k, edge_device_enabled);

    std::unique_ptr<float[]> centroid2queries = std::make_unique<float[]>(n * nprobe);
    std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * nprobe);
    init_result(metric, n * nprobe, centroid2queries.get(), listidqueries.get());

    float* simi = distances;
    idx_t* idxi = labels;
    float* centroids2query = centroid2queries.get();
    idx_t* listids = listidqueries.get();
    // Reused by every PCA signature on this worker; per-list centers overwrite it.
    // Keep baseline/Triangle searches allocation-free when projection pruning is disabled.
    std::vector<float> query_centered;
    if (metric == MetricType::METRIC_L2 && multipivot_mode == MultiPivotMode::PROJECTION) {
        query_centered.resize(d);
    }

    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist,
                                          centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          centroids2query,
                                          listids);
        sort_result(metric, nprobe, centroids2query, listids);

        std::array<double, MULTIPIVOT_MAX_PIVOTS> global_query_pivot_squared{};
        std::array<float, MULTIPIVOT_MAX_PIVOTS> global_query_signature_z{};
        float global_query_sqrt_rho = 0;
        bool global_query_signature_ready = false;
        bool global_full_query_signature_ready = false;
        size_t global_projection_ready_dims = 0;
        size_t global_active_pivots = 0;
        if (metric == MetricType::METRIC_L2 && multipivot_mode != MultiPivotMode::NONE &&
            multipivot_scope == MultiPivotScope::GLOBAL) {
            if (global_pivots.count >= multipivot_active_count &&
                global_pivots.has_U(multipivot_active_count)) {
                global_active_pivots = multipivot_active_count;
            }
#ifdef TRIBASE_ENABLE_STATS
            const auto signature_start = std::chrono::steady_clock::now();
#endif
            const bool use_U = multipivot_mode == MultiPivotMode::PROJECTION &&
                               global_pivots.has_U(global_active_pivots);
            if (use_U && projection_dynamic && projection_block_size != 0 &&
                global_active_pivots > 1) {
                // Dynamic stage-major mode only needs the center radius initially.
                // Projection coordinates are materialized by IVFScan one block at a
                // time when the active candidate set reaches that stage.
                const float* center = global_pivots.codes.data();
                float center_distance_squared = 0.0f;
#pragma omp simd reduction(+ : center_distance_squared)
                for (size_t axis = 0; axis < d; ++axis) {
                    const float delta = queries[i * d + axis] - center[axis];
                    query_centered[axis] = delta;
                    center_distance_squared += delta * delta;
                }
                global_query_pivot_squared[0] = center_distance_squared;
                global_query_signature_ready = true;
            } else if (use_U) {
                global_query_signature_ready = computePcaQuerySignature(
                    queries + i * d, d, global_pivots, global_active_pivots,
                    global_query_pivot_squared.data(), global_query_signature_z.data(),
                    &global_query_sqrt_rho, query_centered.data());
                global_full_query_signature_ready = global_query_signature_ready;
            } else {
                computeQueryPivotDistances(queries + i * d, d, global_pivots,
                                           global_active_pivots,
                                           global_query_pivot_squared.data());
            }
            // Global projection geometry is shared across lists: sign q once per query.
            if (multipivot_mode == MultiPivotMode::PROJECTION &&
                !use_U && global_active_pivots != 0) {
                global_query_signature_ready = computeQuerySignature(
                    global_pivots, global_query_pivot_squared.data(), global_active_pivots,
                    global_query_signature_z.data(), &global_query_sqrt_rho);
            }
            IF_STATS {
                stats->pivot_distance_computations += use_U ? 1 : global_active_pivots;
#ifdef TRIBASE_ENABLE_STATS
                stats->query_signature_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - signature_start)
                        .count();
#endif
            }
        }

        if (metric == MetricType::METRIC_L2) {
            bool previous_dynamic_observation_valid = false;
            double best_log_survival_per_dimension = 0.0;
            for (size_t j = 0; j < nprobe; j++) {
                IVF& list = lists[listids[j]];
                float centroid2query = centroids2query[j];
                size_t list_size = list.get_list_size();
                setTraceContext(stats ? stats->query_offset + i : i, j, listids[j]);

                std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size + 1);
                std::fill_n(if_skip.get(), list_size + 1, false);

                size_t skip_count = 0;
                size_t skip_count_large = 0;
                size_t scan_begin = 0;
                size_t scan_end = list_size;

                if (opt_level & OptLevel::OPT_TRIANGLE) {
                    const float* sqrt_candidate2centroid = list.get_sqrt_candidate2centroid();
                    const float* candidate2centroid = list.get_candidate2centroid();
                    float sqrt_simi = ratio * sqrt(simi[0]);  // TODO:
                    float sqrt_centroid2query = sqrt(centroid2query);
                    for (size_t ii = 0; ii < list_size; ii++) {
                        float tmp = sqrt_simi + sqrt_candidate2centroid[ii];
                        if (tmp < sqrt_centroid2query) {
                            skip_count++;
                        } else {
                            break;
                        }
                    }
                    if (traceEnabled()) {
                        traceLog("stage=triangle_l2_lower,query_id=%zu,probe_rank=%zu,list_id=%lld,list_size=%zu,q2centroid_l2=%.6f,current_topk_radius_l2=%.6f,scan_begin=%zu,pruned=%zu",
                                 trace_query_id, trace_probe_rank, static_cast<long long>(trace_list_id), list_size,
                                 sqrt_centroid2query, sqrt_simi, skip_count, skip_count);
                    }

                    for (int64_t ii = list_size - 1; ii >= 0; ii--) {
                        float tmp_large = sqrt_simi + sqrt_centroid2query;
                        tmp_large *= tmp_large;
                        if (tmp_large < candidate2centroid[ii]) {
                            skip_count_large++;
                        } else {
                            break;
                        }
                    }
                    scan_begin = skip_count;
                    scan_end -= skip_count_large;
                    if (traceEnabled()) {
                        traceLog("stage=triangle_l2_upper,query_id=%zu,probe_rank=%zu,list_id=%lld,list_size=%zu,q2centroid_l2=%.6f,current_topk_radius_l2=%.6f,scan_end=%zu,pruned=%zu,remaining=%zu",
                                 trace_query_id, trace_probe_rank, static_cast<long long>(trace_list_id), list_size,
                                 sqrt_centroid2query, sqrt_simi, scan_end, skip_count_large,
                                 scan_end > scan_begin ? scan_end - scan_begin : 0);
                    }
                }

                IF_STATS {
                    stats->skip_triangle_count += skip_count;
                    stats->skip_triangle_large_count += skip_count_large;
                    stats->total_count += list_size;
                }

                size_t subnn_L2_before = 0, subnn_IP_before = 0, simi_before = 0;
                size_t multipivot_checks_before = 0, multipivot_pruned_before = 0;
                size_t multipivot_invalid_before = 0, candidate_distances_before = 0;
                size_t pivot_distances_before = 0;
                IF_STATS {
                    multipivot_checks_before = stats->multipivot_checks;
                    multipivot_pruned_before = stats->multipivot_pruned;
                    if (stats->collect_list_stats) {
                        subnn_L2_before = stats->skip_subnn_L2_count;
                        subnn_IP_before = stats->skip_subnn_IP_count;
                        simi_before = stats->simi_update_count;
                        multipivot_checks_before = stats->multipivot_checks;
                        multipivot_pruned_before = stats->multipivot_pruned;
                        multipivot_invalid_before = stats->multipivot_invalid;
                        candidate_distances_before = stats->candidate_distance_computations;
                        pivot_distances_before = stats->pivot_distance_computations;
                    }
                }

                // Choose one prefix before scanning the list. Costs are expressed
                // in equivalent scalar-coordinate work:
                //   projection + SIMD LB/residual + predicted surviving exact L2.
                // The previous visited list supplies the observed pruning prior.
                size_t dynamic_pivot_budget = multipivot_active_count;
                if (projection_dynamic && projection_block_size != 0 &&
                    multipivot_mode == MultiPivotMode::PROJECTION) {
                    const size_t max_pivots = multipivot_scope == MultiPivotScope::GLOBAL
                        ? global_active_pivots
                        : multipivot_active_count;
                    const size_t survivors = scan_end > scan_begin ? scan_end - scan_begin : 0;
                    // max+1 is an internal sentinel meaning "run the cost model".
                    dynamic_pivot_budget = max_pivots + 1;
                    if (max_pivots == 0 || survivors == 0) {
                        dynamic_pivot_budget = 0;
                    } else if (multipivot_scope == MultiPivotScope::PER_LIST &&
                               j < static_cast<size_t>(std::ceil(std::sqrt(
                                       static_cast<double>(nprobe))))) {
                        // Per-list projections cannot be reused. Sample a prefix
                        // of the probe order with full P so hard near lists do not
                        // suppress MP before the easier far-list regime is seen.
                        dynamic_pivot_budget = max_pivots;
                    } else if (multipivot_scope == MultiPivotScope::GLOBAL &&
                               max_pivots > 1) {
                        // A small-prefix observation cannot predict whether later
                        // PCA directions become valuable. Probe exponentially
                        // growing prefixes across the first few lists. Higher-D
                        // exact distances justify faster exploration growth. Keep
                        // a 2x floor; 64 dimensions (eight AVX2 float vectors) are
                        // one growth unit. Materialized projections remain reusable.
                        constexpr size_t minimum_exploration_growth_factor = 2;
                        constexpr double dimensions_per_growth_unit = 64.0;
                        const size_t exploration_growth_factor = std::max(
                            minimum_exploration_growth_factor,
                            static_cast<size_t>(std::ceil(std::sqrt(
                                static_cast<double>(d) / dimensions_per_growth_unit))));
                        const double exploration_break_even =
                            static_cast<double>(d) * static_cast<double>(survivors) /
                            (static_cast<double>(d) + static_cast<double>(survivors));
                        const size_t exploration_max_dims = std::min(
                            max_pivots - 1,
                            std::max(projection_block_size,
                                     static_cast<size_t>(exploration_break_even)));
                        // The first prefix must also reflect the amount of exact
                        // work at stake.  m = D*R/(D+R) is the largest number of
                        // projection dimensions whose projection + candidate
                        // update cost can possibly beat R full D-dimensional
                        // distances, even under perfect pruning.  Start at the
                        // square root of that economic range (rounded to a whole
                        // block), rather than always starting at one tiny block.
                        // This gives high-dimensional, populated lists enough
                        // directions to establish useful pruning immediately,
                        // while keeping the first probe small for short lists.
                        const size_t economic_start_dims = std::max<size_t>(
                            1, static_cast<size_t>(std::ceil(
                                   std::sqrt(exploration_break_even))));
                        // A short list cannot amortize a wide exploratory scan.
                        // Limit the prefix to the number of directions supported
                        // by its candidate population.  Scaling R by sqrt(D)
                        // keeps tiny high-dimensional lists at one block without
                        // suppressing useful exploration on populated lists.
                        const size_t occupancy_supported_dims = std::max(
                            projection_block_size,
                            static_cast<size_t>(
                                (2.0 * static_cast<double>(survivors)) /
                                std::sqrt(static_cast<double>(d))));
                        const size_t initial_unaligned_dims = std::min(
                            economic_start_dims, occupancy_supported_dims);
                        const size_t initial_block_count =
                            (initial_unaligned_dims + projection_block_size - 1) /
                            projection_block_size;
                        size_t exploration_dims = std::min(
                            exploration_max_dims,
                            std::max(projection_block_size,
                                     initial_block_count * projection_block_size));
                        for (size_t rank = 0; rank < j &&
                                              exploration_dims < exploration_max_dims; ++rank) {
                            exploration_dims = std::min(
                                exploration_max_dims,
                                exploration_dims >
                                        exploration_max_dims / exploration_growth_factor
                                    ? exploration_max_dims
                                    : exploration_dims * exploration_growth_factor);
                        }
                        const bool exploration_incomplete =
                            j == 0 || global_projection_ready_dims < exploration_max_dims;
                        if (exploration_incomplete &&
                            exploration_dims > global_projection_ready_dims) {
                            dynamic_pivot_budget = exploration_dims + 1;
                        } else if (!previous_dynamic_observation_valid) {
                            dynamic_pivot_budget = std::min(
                                max_pivots, projection_block_size + 1);
                        } else {
                            // Fall through to the cost minimization below after
                            // the geometric exploration reaches its current cap.
                            dynamic_pivot_budget = max_pivots + 1;
                        }
                    } else if (!previous_dynamic_observation_valid) {
                        // Global projections are shared across all visited lists;
                        // materializing full P for exploration would erase all
                        // signature savings. Start with one block instead.
                        dynamic_pivot_budget = multipivot_scope == MultiPivotScope::GLOBAL
                            ? std::min(max_pivots, projection_block_size + 1)
                            : max_pivots;
                    }
                    if (dynamic_pivot_budget > max_pivots) {
                        const double D = static_cast<double>(d);
                        const double R = static_cast<double>(survivors);
                        // One residual sqrt/check costs about six coordinate updates
                        // on the AVX2 kernels used by the current experiments.
                        constexpr double residual_check_cost = 6.0;
                        auto predicted_prune_rate = [&](size_t pivot_budget) {
                            if (pivot_budget == 0) return 0.0;
                            const size_t dims = pivot_budget - 1;
                            if (dims == 0) return 0.0;
                            return std::clamp(
                                1.0 - std::exp(best_log_survival_per_dimension *
                                               static_cast<double>(dims)),
                                              0.0, 0.9999);
                        };
                        auto predicted_cost = [&](size_t pivot_budget) {
                            if (pivot_budget == 0) return R * D;
                            const size_t dims = pivot_budget - 1;
                            const size_t ready_dims = multipivot_scope == MultiPivotScope::GLOBAL
                                ? global_projection_ready_dims : 0;
                            const size_t new_query_dims = dims > ready_dims ? dims - ready_dims : 0;
                            const double projection_cost =
                                static_cast<double>(new_query_dims) * D;
                            const double lower_bound_cost =
                                R * (static_cast<double>(dims) + residual_check_cost);
                            const double exact_cost =
                                R * (1.0 - predicted_prune_rate(pivot_budget)) * D;
                            return projection_cost + lower_bound_cost + exact_cost;
                        };

                        dynamic_pivot_budget = 0;
                        double best_cost = predicted_cost(0);
                        // P=1 is center-only. Limit the budget search to roughly
                        // twenty proportional samples when max P is large, while
                        // retaining a minimum two-direction/configured-block step.
                        // The exact configured P is evaluated separately below.
                        constexpr size_t target_budget_samples = 20;
                        const size_t max_projection_dims = max_pivots - 1;
                        const size_t proportional_step =
                            (max_projection_dims + target_budget_samples - 1) /
                            target_budget_samples;
                        const size_t budget_step = std::max(
                            {size_t{2}, projection_block_size, proportional_step});
                        for (size_t dims = 0; dims + 1 < max_pivots;
                             dims += budget_step) {
                            const size_t candidate_budget = std::min(max_pivots, dims + 1);
                            const double cost = predicted_cost(candidate_budget);
                            if (cost < best_cost) {
                                best_cost = cost;
                                dynamic_pivot_budget = candidate_budget;
                            }
                        }
                        const double full_cost = predicted_cost(max_pivots);
                        if (full_cost < best_cost) dynamic_pivot_budget = max_pivots;
                    }
                }

                std::array<double, MULTIPIVOT_MAX_PIVOTS> per_list_query_pivot_squared{};
                std::array<float, MULTIPIVOT_MAX_PIVOTS> per_list_query_signature_z{};
                size_t per_list_projection_ready_dims = 0;
                float per_list_query_sqrt_rho = 0;
                bool per_list_query_signature_ready = false;
                const PivotMetadata* pivot_metadata = nullptr;
                const double* query_pivot_squared = nullptr;
                float* query_signature_z = nullptr;
                float query_sqrt_rho = 0;
                bool query_signature_ready = false;
                size_t active_pivots = 0;
                MultiPivotMode scan_multipivot_mode = multipivot_mode;
                const size_t configured_full_pivots =
                    multipivot_scope == MultiPivotScope::GLOBAL
                        ? global_active_pivots
                        : std::min(multipivot_active_count, list.pivots.count);
                const bool use_dynamic_prefix_for_list =
                    projection_dynamic && projection_block_size != 0 &&
                    dynamic_pivot_budget != 0 &&
                    dynamic_pivot_budget < configured_full_pivots;
                if (multipivot_mode != MultiPivotMode::NONE) {
                    pivot_metadata = &list.pivots;
                    if (multipivot_scope == MultiPivotScope::GLOBAL) {
                        active_pivots = dynamic_pivot_budget == multipivot_active_count
                            ? global_active_pivots
                            : std::min(global_active_pivots, dynamic_pivot_budget);
                        // Per-list metadata owns the candidate signatures, while the
                        // global metadata owns the shared PCA basis. Materialize any
                        // newly requested global prefix here so IVFScan never tries
                        // to read U from a per-list signature container.
                        if (use_dynamic_prefix_for_list &&
                            active_pivots > 1 && global_pivots.has_U(active_pivots)) {
                            const size_t requested_dims = active_pivots - 1;
#ifdef TRIBASE_ENABLE_STATS
                            const auto projection_start = std::chrono::steady_clock::now();
#endif
                            for (size_t dim = global_projection_ready_dims;
                                 dim < requested_dims; ++dim) {
                                float qz = 0.0f;
#pragma omp simd reduction(+ : qz)
                                for (size_t axis = 0; axis < d; ++axis) {
                                    qz += query_centered[axis] *
                                          global_pivots.U[dim * d + axis];
                                }
                                global_query_signature_z[dim] = qz;
                            }
                            global_projection_ready_dims =
                                std::max(global_projection_ready_dims, requested_dims);
#ifdef TRIBASE_ENABLE_STATS
                            if (stats != nullptr) {
                                stats->query_signature_seconds +=
                                    std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - projection_start)
                                        .count();
                            }
#endif
                        }
                        if (!use_dynamic_prefix_for_list && active_pivots > 1 &&
                            !global_full_query_signature_ready &&
                            global_pivots.has_U(active_pivots)) {
#ifdef TRIBASE_ENABLE_STATS
                            const auto full_signature_start = std::chrono::steady_clock::now();
#endif
                            global_full_query_signature_ready = computePcaQuerySignature(
                                queries + i * d, d, global_pivots, active_pivots,
                                global_query_pivot_squared.data(),
                                global_query_signature_z.data(), &global_query_sqrt_rho,
                                query_centered.data());
                            global_query_signature_ready = global_full_query_signature_ready;
                            global_projection_ready_dims = active_pivots - 1;
#ifdef TRIBASE_ENABLE_STATS
                            if (stats != nullptr) {
                                stats->query_signature_seconds +=
                                    std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - full_signature_start)
                                        .count();
                            }
#endif
                        }
                        query_pivot_squared = global_query_pivot_squared.data();
                        query_signature_z = global_query_signature_z.data();
                        query_sqrt_rho = global_query_sqrt_rho;
                        query_signature_ready = global_query_signature_ready;
                    } else {
                        if (list.pivots.count >= dynamic_pivot_budget && dynamic_pivot_budget != 0) {
                            active_pivots = std::min(list.pivots.count, dynamic_pivot_budget);
                        } else if (!projection_dynamic) {
                            active_pivots = list.pivots.count >= multipivot_active_count &&
                                            list.pivots.has_U(multipivot_active_count)
                                ? multipivot_active_count : 0;
                        }
#ifdef TRIBASE_ENABLE_STATS
                        const auto signature_start = std::chrono::steady_clock::now();
#endif
                        const bool use_U = multipivot_mode == MultiPivotMode::PROJECTION &&
                                           list.pivots.has_U(active_pivots);
                        if (use_U && use_dynamic_prefix_for_list &&
                            active_pivots > 1) {
                            // For centroid-anchored per-list PCA, pivot 0 is exactly
                            // the IVF centroid. coarse search has already computed
                            // this squared distance for the current probe.
                            per_list_query_pivot_squared[0] = centroid2query;
                            per_list_query_signature_ready = true;
                        } else if (use_U) {
                            per_list_query_signature_ready = computePcaQuerySignature(
                                queries + i * d, d, list.pivots, active_pivots,
                                per_list_query_pivot_squared.data(),
                                per_list_query_signature_z.data(),
                                &per_list_query_sqrt_rho, query_centered.data());
                        } else {
                            computeQueryPivotDistances(
                                queries + i * d, d, list.pivots, active_pivots,
                                per_list_query_pivot_squared.data());
                        }
                        query_pivot_squared = per_list_query_pivot_squared.data();
                        // Per-list projection geometry differs: sign q once for this list.
                        if (multipivot_mode == MultiPivotMode::PROJECTION &&
                            !use_U && active_pivots != 0) {
                            per_list_query_signature_ready = computeQuerySignature(
                                list.pivots, query_pivot_squared, active_pivots,
                                per_list_query_signature_z.data(), &per_list_query_sqrt_rho);
                        }
                        query_signature_z = per_list_query_signature_z.data();
                        query_sqrt_rho = per_list_query_sqrt_rho;
                        query_signature_ready = per_list_query_signature_ready;
                        IF_STATS {
                            stats->pivot_distance_computations += use_U ? 1 : active_pivots;
#ifdef TRIBASE_ENABLE_STATS
                            stats->query_signature_seconds +=
                                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                              signature_start)
                                    .count();
#endif
                        }
                    }
                }
                IF_STATS {
                    stats->active_pivot_histogram[
                        std::min(active_pivots, stats->active_pivot_histogram.size() - 1)]++;
                }
                // A zero list-level budget is an intentional baseline decision:
                // disable the MP dispatch completely instead of entering the
                // generic per-candidate MP path with no usable metadata.
                if (projection_dynamic && dynamic_pivot_budget == 0) {
                    scan_multipivot_mode = MultiPivotMode::NONE;
                }

                // Microblock AABB prune in the persisted P-dimensional Φ space.
                // Marks if_skip; no disk I/O — used to measure block-level prune rate.
                if (microblock_enabled && metric == MetricType::METRIC_L2 &&
                    !list.microblocks.empty()) {
                    const size_t built_P =
                        list.pivots.count == multipivot_built_count ? multipivot_built_count : 0;
                    if (built_P >= 1 && multipivot_active_count == built_P &&
                        list.pivots.usable_signature(built_P)) {
                        std::array<double, MULTIPIVOT_MAX_PIVOTS> mb_query_pivot_squared{};
                        std::array<float, MULTIPIVOT_MAX_PIVOTS> mb_z{};
                        float mb_sqrt_rho = 0;
                        bool mb_ready = false;
                        if (active_pivots == built_P && query_signature_ready &&
                            query_signature_z != nullptr) {
                            for (size_t t = 0; t + 1 < built_P; ++t) mb_z[t] = query_signature_z[t];
                            mb_sqrt_rho = query_sqrt_rho;
                            mb_ready = true;
                        } else if (list.pivots.has_U(built_P)) {
                            mb_ready = computePcaQuerySignature(
                                queries + i * d, d, list.pivots, built_P,
                                mb_query_pivot_squared.data(), mb_z.data(), &mb_sqrt_rho,
                                query_centered.data());
                        } else {
                            computeQueryPivotDistances(queries + i * d, d, list.pivots, built_P,
                                                       mb_query_pivot_squared.data());
                            mb_ready = computeQuerySignature(list.pivots, mb_query_pivot_squared.data(),
                                                             built_P, mb_z.data(), &mb_sqrt_rho);
                        }
                        if (mb_ready) {
                            std::array<float, MULTIPIVOT_MAX_PIVOTS> phi_q{};
                            if (built_P == 1) {
                                phi_q[0] = mb_sqrt_rho;
                            } else {
                                for (size_t t = 0; t + 1 < built_P; ++t) phi_q[t] = mb_z[t];
                                phi_q[built_P - 1] = mb_sqrt_rho;
                            }
                            const float radius = simi[0];
                            for (const MicroblockMeta& block : list.microblocks) {
                                if (block.lower.size() != built_P ||
                                    block.upper.size() != built_P) {
                                    continue;
                                }
                                IF_STATS { stats->microblock_checks++; }
                                const float lb2 = aabbLowerBoundSquared(
                                    phi_q.data(), block.lower.data(), block.upper.data(), built_P);
                                if (lb2 > radius) {
                                    size_t vectors_pruned = 0;
                                    for (uint32_t member : block.members) {
                                        if (member >= scan_begin && member < scan_end &&
                                            !if_skip[member]) {
                                            if_skip[member] = true;
                                            ++vectors_pruned;
                                        }
                                    }
                                    IF_STATS {
                                        stats->microblock_pruned++;
                                        stats->microblock_vectors_pruned += vectors_pruned;
                                    }
                                }
                            }
                        }
                    }
                }

#ifdef TRIBASE_ENABLE_STATS
                const auto candidate_decision_start = std::chrono::steady_clock::now();
#endif
                scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(), list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
                                   list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                                   list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(), list.get_sub_farest_IP_dis(),
                                   list.get_sub_nearest_L2_id(), list.get_sub_nearest_L2_dis(),
                                   pivot_metadata, query_pivot_squared, active_pivots, scan_multipivot_mode,
                                   query_signature_z, query_sqrt_rho, projection_prefix_length,
                                   use_dynamic_prefix_for_list ? projection_block_size : 0,
                                   use_dynamic_prefix_for_list,
                                   multipivot_scope == MultiPivotScope::GLOBAL
                                       ? &global_projection_ready_dims
                                       : &per_list_projection_ready_dims,
                                   query_signature_ready,
                                   if_skip.get(), simi, idxi,
                                   stats, centroid_codes.get() + listids[j] * d, ratio, ratio);
                if (projection_dynamic && scaner->last_multipivot_checks != 0) {
                    const double observed_prune_rate =
                        static_cast<double>(scaner->last_multipivot_pruned) /
                        static_cast<double>(scaner->last_multipivot_checks);
                    if (active_pivots > 1) {
                        const double observed_log_survival_per_dimension =
                            std::log(std::clamp(1.0 - observed_prune_rate, 1e-4, 1.0)) /
                            static_cast<double>(active_pivots - 1);
                        // Probe ranks move away from the query centroid and usually
                        // become easier to prune. Preserve the strongest prior seen
                        // so an early hard list cannot permanently disable MP.
                        if (!previous_dynamic_observation_valid) {
                            best_log_survival_per_dimension =
                                observed_log_survival_per_dimension;
                        } else {
                            best_log_survival_per_dimension = std::min(
                                best_log_survival_per_dimension,
                                observed_log_survival_per_dimension);
                        }
                        previous_dynamic_observation_valid = true;
                    }
                }
                IF_STATS {
                    const size_t checks_delta = stats->multipivot_checks - multipivot_checks_before;
                    const size_t pruned_delta = stats->multipivot_pruned - multipivot_pruned_before;
#ifdef TRIBASE_ENABLE_STATS
                    stats->candidate_decision_seconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                      candidate_decision_start)
                            .count();
#endif
                }

                IF_STATS {
                    if (stats->collect_list_stats) {
                        ListVisitStat visit;
                        visit.query_id = stats->query_offset + i;
                        visit.probe_rank = j;
                        visit.list_id = listids[j];
                        visit.list_size = list_size;
                        visit.centroid2query = centroid2query;
                        visit.scan_begin = scan_begin;
                        visit.scan_end = scan_end;
                        visit.tri = skip_count;
                        visit.tri_large = skip_count_large;
                        visit.subnn_L2 = stats->skip_subnn_L2_count - subnn_L2_before;
                        visit.subnn_IP = stats->skip_subnn_IP_count - subnn_IP_before;
                        visit.multipivot_checks = stats->multipivot_checks - multipivot_checks_before;
                        visit.multipivot_pruned = stats->multipivot_pruned - multipivot_pruned_before;
                        visit.multipivot_invalid = stats->multipivot_invalid - multipivot_invalid_before;
                        visit.active_pivots = active_pivots;
                        visit.candidate_distances =
                            stats->candidate_distance_computations - candidate_distances_before;
                        visit.pivot_distances =
                            stats->pivot_distance_computations - pivot_distances_before;
                        visit.simi_update = stats->simi_update_count - simi_before;
                        stats->list_visits.push_back(visit);
                    }
                }
            }
        } else {
            for (size_t j = 0; j < nprobe; j++) {
                IVF& list = lists[listids[j]];
                const float* candidate2centroid = list.get_candidate2centroid();
                float centroid2query = centroids2query[j];
                setTraceContext(stats ? stats->query_offset + i : i, j, listids[j]);
                float s_centroid2query = sqrt(1 - centroid2query * centroid2query);
                float s_simi = sqrt(1 - simi[0] * simi[0]);
                size_t list_size = list.get_list_size();
                size_t scan_begin = 0;
                size_t scan_end = list_size;
                if (opt_level & OptLevel::OPT_TRIANGLE) {
                    float min_cut_degree_cos;
                    float max_cut_degree_cos;
                    if (simi[0] < centroid2query) {  // 0 ~ c + s
                        max_cut_degree_cos = 1;
                        min_cut_degree_cos = simi[0] * centroid2query - s_simi * s_centroid2query;
                        while (scan_begin < scan_end && candidate2centroid[scan_end - 1] < min_cut_degree_cos) {
                            scan_end--;
                        }
                    } else {  // c - s ~ c + s
                        max_cut_degree_cos = simi[0] * centroid2query + s_simi * s_centroid2query;
                        min_cut_degree_cos = simi[0] * centroid2query - s_simi * s_centroid2query;
                        while (scan_begin < scan_end && candidate2centroid[scan_begin] > max_cut_degree_cos) {
                            scan_begin++;
                        }
                        while (scan_begin < scan_end && candidate2centroid[scan_end - 1] < min_cut_degree_cos) {
                            scan_end--;
                        }
                    }
                    if (traceEnabled()) {
                        traceLog("stage=triangle_ip_range,query_id=%zu,probe_rank=%zu,list_id=%lld,list_size=%zu,q2centroid_ip=%.6f,current_topk_ip=%.6f,min_cut_cos=%.6f,max_cut_cos=%.6f,scan_begin=%zu,scan_end=%zu,pruned=%zu",
                                 trace_query_id, trace_probe_rank, static_cast<long long>(trace_list_id), list_size,
                                 centroid2query, simi[0], min_cut_degree_cos, max_cut_degree_cos,
                                 scan_begin, scan_end, scan_begin + list_size - scan_end);
                    }
                    IF_STATS {
                        stats->skip_triangle_count += scan_begin + list_size - scan_end;
                        stats->total_count += list_size;
                    }
                }
#ifdef TRIBASE_ENABLE_STATS
                const auto candidate_decision_start = std::chrono::steady_clock::now();
#endif
                scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(),
                                   list.get_candidate_id(), simi, idxi);
                IF_STATS {
#ifdef TRIBASE_ENABLE_STATS
                    stats->candidate_decision_seconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                      candidate_decision_start)
                            .count();
#endif
                }

                IF_STATS {
                    if (stats->collect_list_stats) {
                        if (!(opt_level & OptLevel::OPT_TRIANGLE)) {
                            stats->total_count += list_size;
                        }
                        ListVisitStat visit;
                        visit.query_id = stats->query_offset + i;
                        visit.probe_rank = j;
                        visit.list_id = listids[j];
                        visit.list_size = list_size;
                        visit.centroid2query = centroid2query;
                        visit.scan_begin = scan_begin;
                        visit.scan_end = scan_end;
                        visit.tri = (opt_level & OptLevel::OPT_TRIANGLE) ? (scan_begin + list_size - scan_end) : 0;
                        visit.tri_large = 0;
                        visit.subnn_L2 = 0;
                        visit.subnn_IP = 0;
                        visit.simi_update = 0;
                        stats->list_visits.push_back(visit);
                    }
                }
            }
        }
        sort_result(metric, k, simi, idxi);

        simi += k;
        idxi += k;
        centroids2query += nprobe;
        listids += nprobe;
    }
#ifdef TRIBASE_ENABLE_STATS
    if (stats) {
        stats->search_worker_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - worker_search_start)
                .count();
    }
#endif
}

Stats Index::search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio) {
    if (n == 0) {
        return Stats();
    }
    if ((opt_level & added_opt_level) != opt_level) {
        std::cerr << "opt_level: " << opt_level << " added_opt_level: " << added_opt_level << std::endl;
        throw std::runtime_error("opt_level is not subset of added_opt_level");
    }
    if (nprobe > nlist) {
        nprobe = nlist;
    }
    init_result(metric, n * k, distances, labels);
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;
    size_t extra = n % nt;
    std::vector<Stats> stats(nt);
    for (size_t i = 0; i < nt; i++) {
        stats[i].collect_list_stats = collect_list_stats;
    }

#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            stats[i].query_offset = start;
            if (stats[i].collect_list_stats) {
                stats[i].list_visits.reserve((end - start) * nprobe);
            }
            single_thread_search(end - start, queries + start * d, k, distances + start * k, labels + start * k, ratio, &stats[i]);
        }
    }

    [[maybe_unused]] Stats total_stats = mergeStats(stats);
    total_stats.collect_list_stats = collect_list_stats;
    return total_stats;
}

void Index::save_index(std::string path) const {
    prepareDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    out.write(INDEX_MAGIC, sizeof(INDEX_MAGIC));
    const uint32_t version = INDEX_FORMAT_VERSION;
    const uint32_t endian = INDEX_ENDIAN_MARKER;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&endian), sizeof(endian));
    out.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&nprobe), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(MetricType));
    out.write(reinterpret_cast<const char*>(&added_opt_level), sizeof(OptLevel));
    out.write(reinterpret_cast<const char*>(&sub_k), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&sub_nlist), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&sub_nprobe), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&multipivot_scope), sizeof(multipivot_scope));
    out.write(reinterpret_cast<const char*>(&multipivot_built_count), sizeof(multipivot_built_count));
    out.write(reinterpret_cast<const char*>(&multipivot_seed), sizeof(multipivot_seed));
    const uint32_t method_length = multipivot_method.size();
    out.write(reinterpret_cast<const char*>(&method_length), sizeof(method_length));
    out.write(multipivot_method.data(), method_length);
    out.write(reinterpret_cast<const char*>(&multipivot_selection_seconds), sizeof(multipivot_selection_seconds));
    out.write(reinterpret_cast<const char*>(&multipivot_candidate_distance_seconds), sizeof(multipivot_candidate_distance_seconds));
    out.write(reinterpret_cast<const char*>(&multipivot_build_distance_computations), sizeof(multipivot_build_distance_computations));
    global_pivots.save(out);

    out.write(reinterpret_cast<const char*>(centroid_codes.get()), nlist * d * sizeof(float));
    // out.write(reinterpret_cast<const char*>(centroid_ids.get()), nlist * sizeof(idx_t)); // 0 ~ nlist-1

    uint64_t nonempty_lists = 0;
    for (size_t i = 0; i < nlist; ++i) nonempty_lists += lists[i].get_list_size() != 0;
    out.write(reinterpret_cast<const char*>(&nonempty_lists), sizeof(nonempty_lists));
    for (size_t i = 0; i < nlist; i++) {
        if (lists[i].get_list_size() > 0) {
            out.write(reinterpret_cast<const char*>(&i), sizeof(size_t));
            lists[i].save_IVF(out);
        }
    }
    if (!out) throw std::runtime_error("Failed while writing index " + path);
}

void Index::load_legacy_index(std::string path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&metric), sizeof(MetricType));
    in.read(reinterpret_cast<char*>(&added_opt_level), sizeof(OptLevel));
    opt_level = OptLevel::OPT_NONE;
    in.read(reinterpret_cast<char*>(&sub_k), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nprobe), sizeof(size_t));
    if (!in || d == 0 || nlist == 0 || nlist > (1ULL << 32)) {
        throw std::runtime_error("Invalid legacy Tribase index metadata");
    }

    multipivot_scope = MultiPivotScope::GLOBAL;
    multipivot_mode = MultiPivotMode::NONE;
    multipivot_method = "affine_fps";
    multipivot_built_count = 0;
    multipivot_active_count = 0;
    multipivot_seed = 0;
    multipivot_selection_seconds = 0;
    multipivot_candidate_distance_seconds = 0;
    multipivot_build_distance_computations = 0;
    global_pivots.clear();

    centroid_codes = std::make_unique<float[]>(nlist * d);
    in.read(reinterpret_cast<char*>(centroid_codes.get()), nlist * d * sizeof(float));
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);

    lists = std::make_unique<IVF[]>(nlist);
    Stopwatch watch;
    while (true) {
        size_t listid;
        in.read(reinterpret_cast<char*>(&listid), sizeof(size_t));
        if (!in) {
            if (in.eof()) break;
            throw std::runtime_error("Failed while reading legacy IVF list header");
        }
        if (listid >= nlist || lists[listid].get_list_size() != 0) {
            throw std::runtime_error("Invalid or duplicate IVF list ID in legacy index");
        }
        lists[listid].load_IVF_legacy(in);
        if (!in) {
            throw std::runtime_error("Failed while reading legacy IVF list body");
        }
        if (watch.elapsedSeconds() > 2) {
            watch.reset();
            std::cout << std::format("loaded legacy list {}/{}, size {}", listid, nlist,
                                     lists[listid].get_list_size())
                      << std::endl;
        }
    }
}

void Index::load_index(std::string path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    char magic[sizeof(INDEX_MAGIC)]{};
    in.read(magic, sizeof(magic));
    uint32_t version = 0;
    uint32_t endian = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&endian), sizeof(endian));
    if (!in || std::memcmp(magic, INDEX_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error(std::format(
            "Legacy or invalid Tribase index; rebuild with format version {}",
            INDEX_FORMAT_VERSION));
    }
    if (version != INDEX_FORMAT_VERSION || endian != INDEX_ENDIAN_MARKER) {
        throw std::runtime_error("Unsupported Tribase index format/version or endianness");
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(size_t));
    // in.read(reinterpret_cast<char*>(&nprobe), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&metric), sizeof(MetricType));
    in.read(reinterpret_cast<char*>(&added_opt_level), sizeof(OptLevel));
    opt_level = OptLevel::OPT_NONE;
    in.read(reinterpret_cast<char*>(&sub_k), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nprobe), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&multipivot_scope), sizeof(multipivot_scope));
    in.read(reinterpret_cast<char*>(&multipivot_built_count), sizeof(multipivot_built_count));
    in.read(reinterpret_cast<char*>(&multipivot_seed), sizeof(multipivot_seed));
    uint32_t method_length = 0;
    in.read(reinterpret_cast<char*>(&method_length), sizeof(method_length));
    if (!in || method_length > 64) throw std::runtime_error("Invalid multipivot method metadata");
    multipivot_method.resize(method_length);
    in.read(multipivot_method.data(), method_length);
    in.read(reinterpret_cast<char*>(&multipivot_selection_seconds), sizeof(multipivot_selection_seconds));
    in.read(reinterpret_cast<char*>(&multipivot_candidate_distance_seconds), sizeof(multipivot_candidate_distance_seconds));
    in.read(reinterpret_cast<char*>(&multipivot_build_distance_computations), sizeof(multipivot_build_distance_computations));
    global_pivots.load(in, d, 0);
    if (d == 0 || nlist == 0 || nlist > (1ULL << 32) ||
        multipivot_built_count > MULTIPIVOT_MAX_PIVOTS ||
        (multipivot_built_count != 0 && !validMultipivotMethod(multipivot_method))) {
        throw std::runtime_error("Invalid Tribase index metadata");
    }
    multipivot_mode = MultiPivotMode::NONE;
    multipivot_active_count = 0;

    centroid_codes = std::make_unique<float[]>(nlist * d);
    in.read(reinterpret_cast<char*>(centroid_codes.get()), nlist * d * sizeof(float));
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);

    lists = std::make_unique<IVF[]>(nlist);
    Stopwatch watch;
    uint64_t nonempty_lists = 0;
    in.read(reinterpret_cast<char*>(&nonempty_lists), sizeof(nonempty_lists));
    if (!in || nonempty_lists > nlist) throw std::runtime_error("Invalid IVF list count in index");
    for (uint64_t record = 0; record < nonempty_lists; ++record) {
        size_t listid;
        in.read(reinterpret_cast<char*>(&listid), sizeof(size_t));
        if (!in || listid >= nlist || lists[listid].get_list_size() != 0) {
            throw std::runtime_error("Invalid or duplicate IVF list ID in index");
        }
        lists[listid].load_IVF(in);
        if (watch.elapsedSeconds() > 2) {
            watch.reset();
            std::cout << std::format("loaded list {}/{}, size {}", listid, nlist, lists[listid].get_list_size()) << std::endl;
        }
    }
    if (in.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Unexpected trailing data in Tribase index");
    }
}

void Index::probe_lower_bounds(size_t n,
                               const float* queries,
                               const float* tau2_per_query,
                               const std::string& raw_csv_path,
                               const LbProbeConfig& config) const {
    if (metric != MetricType::METRIC_L2) {
        throw std::runtime_error("probe_lower_bounds is L2-only");
    }
    if (!lists || !centroid_codes || nlist == 0 || d == 0) {
        throw std::runtime_error("probe_lower_bounds requires a loaded IVF index");
    }
    if (multipivot_built_count == 0) {
        throw std::runtime_error("probe_lower_bounds requires multipivot metadata (built_count>0)");
    }
    if (n == 0 || queries == nullptr || tau2_per_query == nullptr) {
        throw std::invalid_argument("probe_lower_bounds: empty queries/tau2");
    }
    if (raw_csv_path.empty()) {
        throw std::invalid_argument("probe_lower_bounds: raw_csv_path is empty");
    }

    const size_t P = multipivot_built_count;
    const size_t probes = std::max<size_t>(1, std::min(config.max_probes == 0 ? nprobe : config.max_probes, nprobe));
    const size_t max_per_list =
        config.max_candidates_per_list == 0 ? std::numeric_limits<size_t>::max()
                                           : config.max_candidates_per_list;
    const float alpha = std::max(0.0f, config.boundary_alpha);
    const double hard_scale = static_cast<double>((1.0f + alpha) * (1.0f + alpha));

    struct Agg {
        uint64_t n = 0;
        uint64_t n_near = 0;
        uint64_t n_hard = 0;
        uint64_t n_far = 0;
        uint64_t n_neg = 0;          // d2 > tau2
        uint64_t n_prune = 0;        // lb2 > tau2 among negatives
        double sum_rho = 0;
        double sum_rho_hard = 0;
        double sum_lb2 = 0;
        double sum_d2 = 0;
        double sum_gap_pos = 0;      // E[max(0, lb2-tau2)] on negatives
        double sum_gap_neg = 0;      // E[max(0, tau2-lb2)] on negatives (under-prune residual)
    };
    Agg agg;

    prepareDirectory(raw_csv_path);
    const std::string summary_path = raw_csv_path + ".summary.csv";
    const bool write_raw_header =
        !config.append_raw || !std::filesystem::exists(raw_csv_path) ||
        std::filesystem::file_size(raw_csv_path) == 0;
    std::ofstream raw_out(raw_csv_path, config.append_raw ? std::ios::app : std::ios::trunc);
    if (!raw_out) throw std::runtime_error("Cannot open lb probe csv: " + raw_csv_path);
    if (write_raw_header) {
        raw_out << "q_id,list_id,cand_pos,cand_id,P,d2,lb2,tau2,rho,band,tri_survives,"
                   "lb_gt_tau,scope,method\n";
    }

    std::atomic<uint64_t> raw_rows{0};
    std::mutex raw_mu;
    const uint64_t raw_cap =
        config.max_raw_rows == 0 ? std::numeric_limits<uint64_t>::max() : config.max_raw_rows;

    const std::string scope_name =
        multipivot_scope == MultiPivotScope::GLOBAL ? "global" : "per_list";

#pragma omp parallel
    {
        Agg local_agg;
        std::unique_ptr<IVFScanBase> scaner_quantizer =
            // const_cast: get_scanner is non-const but only reads type tags.
            const_cast<Index*>(this)->get_scanner(metric, OPT_NONE, probes, edge_device_enabled);
        std::vector<float> centroid2queries(probes);
        std::vector<idx_t> listids(probes);
        std::array<double, MULTIPIVOT_MAX_PIVOTS> q_pivot_sq{};
        std::array<float, MULTIPIVOT_MAX_PIVOTS> q_z{};
        std::vector<float> query_centered(d);
        std::string local_raw;
        local_raw.reserve(1 << 16);

#pragma omp for schedule(dynamic, 1)
        for (size_t qi = 0; qi < n; ++qi) {
            const float* q = queries + qi * d;
            const double tau2 = static_cast<double>(tau2_per_query[qi]);
            if (!(tau2 > 0.0) || !std::isfinite(tau2)) continue;
            const double hard_hi = tau2 * hard_scale;
            const float sqrt_tau = static_cast<float>(std::sqrt(tau2));

            init_result(metric, probes, centroid2queries.data(), listids.data());
            scaner_quantizer->set_query(q);
            scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(),
                                             reinterpret_cast<const size_t*>(centroid_ids.get()),
                                             centroid2queries.data(), listids.data());
            sort_result(metric, probes, centroid2queries.data(), listids.data());

            // Global projection geometry is shared across lists, so sign once per query.
            std::array<double, MULTIPIVOT_MAX_PIVOTS> global_q_pivot{};
            std::array<float, MULTIPIVOT_MAX_PIVOTS> global_q_z{};
            float global_sqrt_rho = 0;
            bool global_signature_ready = false;
            if (multipivot_scope == MultiPivotScope::GLOBAL && global_pivots.count == P) {
                if (global_pivots.has_U(P)) {
                    global_signature_ready = computePcaQuerySignature(
                        q, d, global_pivots, P, global_q_pivot.data(), global_q_z.data(),
                        &global_sqrt_rho, query_centered.data());
                } else {
                    computeQueryPivotDistances(q, d, global_pivots, P,
                                               global_q_pivot.data());
                    global_signature_ready = computeQuerySignature(
                        global_pivots, global_q_pivot.data(), P, global_q_z.data(),
                        &global_sqrt_rho);
                }
            }

            for (size_t pj = 0; pj < probes; ++pj) {
                const idx_t list_id = listids[pj];
                if (list_id < 0 || static_cast<size_t>(list_id) >= nlist) continue;
                const IVF& list = lists[list_id];
                const size_t list_size = list.list_size;
                if (list_size == 0) continue;
                const PivotMetadata& meta = list.pivots;
                const size_t built_P = meta.count == P ? P : 0;
                if (built_P == 0 || !meta.usable_signature(built_P)) continue;

                const float centroid2query = centroid2queries[pj];
                const float sqrt_qc = std::sqrt(std::max(0.0f, centroid2query));
                const bool have_tri =
                    list.get_sqrt_candidate2centroid() != nullptr && list.get_candidate2centroid() != nullptr;

                float sqrt_rho_q = 0;
                bool query_signature_ready = false;
                if (multipivot_scope == MultiPivotScope::GLOBAL) {
                    if (!global_signature_ready) continue;
                    std::copy_n(global_q_z.data(), built_P > 0 ? built_P - 1 : 0, q_z.data());
                    sqrt_rho_q = global_sqrt_rho;
                    query_signature_ready = true;
                } else if (meta.has_U(built_P)) {
                    query_signature_ready = computePcaQuerySignature(
                        q, d, meta, built_P, q_pivot_sq.data(), q_z.data(), &sqrt_rho_q,
                        query_centered.data());
                } else {
                    computeQueryPivotDistances(q, d, meta, built_P, q_pivot_sq.data());
                    query_signature_ready = computeQuerySignature(
                        meta, q_pivot_sq.data(), built_P, q_z.data(), &sqrt_rho_q);
                }
                if (!query_signature_ready) continue;

                const size_t stride =
                    max_per_list >= list_size ? 1 : (list_size + max_per_list - 1) / max_per_list;
                size_t taken = 0;
                for (size_t ci = 0; ci < list_size && taken < max_per_list; ci += stride, ++taken) {
                    bool tri_survives = true;
                    if (have_tri) {
                        const float r = list.get_sqrt_candidate2centroid(ci);
                        const float r2 = list.get_candidate2centroid(ci);
                        if (sqrt_tau + r < sqrt_qc) tri_survives = false;
                        else {
                            const float upper = sqrt_tau + sqrt_qc;
                            if (upper * upper < r2) tri_survives = false;
                        }
                    }
                    if (config.triangle_survivors_only && !tri_survives) continue;

                    const float* x = list.get_candidate_codes(ci);
                    const double d2 = static_cast<double>(calculatedEuclideanDistance(q, x, d));
                    if (!std::isfinite(d2) || d2 < 0.0) continue;

                    int band = 2;  // far
                    if (d2 <= tau2) band = 0;
                    else if (d2 <= hard_hi) band = 1;

                    const float lb2f =
                        signatureLowerBoundSquared(q_z.data(), sqrt_rho_q, meta, ci, built_P);
                    const double lb2 = static_cast<double>(lb2f);
                    if (!std::isfinite(lb2)) continue;
                    const double rho =
                        d2 > 1e-12 ? std::min(1.0, std::max(0.0, lb2 / d2)) : 0.0;
                    const bool is_neg = d2 > tau2;
                    const bool prune = lb2 > tau2;

                    Agg& a = local_agg;
                    a.n++;
                    a.sum_rho += rho;
                    a.sum_lb2 += lb2;
                    a.sum_d2 += d2;
                    if (band == 0) a.n_near++;
                    else if (band == 1) {
                        a.n_hard++;
                        a.sum_rho_hard += rho;
                    } else {
                        a.n_far++;
                    }
                    if (is_neg) {
                        a.n_neg++;
                        if (prune) {
                            a.n_prune++;
                            a.sum_gap_pos += lb2 - tau2;
                        } else {
                            a.sum_gap_neg += tau2 - lb2;
                        }
                    }
                    if (raw_rows.load(std::memory_order_relaxed) < raw_cap) {
                        const uint64_t before = raw_rows.fetch_add(1, std::memory_order_relaxed);
                        if (before < raw_cap) {
                            local_raw += std::to_string(qi);
                            local_raw += ',';
                            local_raw += std::to_string(static_cast<long long>(list_id));
                            local_raw += ',';
                            local_raw += std::to_string(ci);
                            local_raw += ',';
                            local_raw += std::to_string(list.get_candidate_id(ci));
                            local_raw += ',';
                            local_raw += std::to_string(built_P);
                            local_raw += ',';
                            local_raw += std::to_string(d2);
                            local_raw += ',';
                            local_raw += std::to_string(lb2);
                            local_raw += ',';
                            local_raw += std::to_string(tau2);
                            local_raw += ',';
                            local_raw += std::to_string(rho);
                            local_raw += ',';
                            local_raw += (band == 0 ? "near" : band == 1 ? "hard" : "far");
                            local_raw += ',';
                            local_raw += (tri_survives ? '1' : '0');
                            local_raw += ',';
                            local_raw += (prune ? '1' : '0');
                            local_raw += ',';
                            local_raw += scope_name;
                            local_raw += ',';
                            local_raw += multipivot_method;
                            local_raw += '\n';
                            if (local_raw.size() > (1u << 15)) {
                                std::lock_guard<std::mutex> lock(raw_mu);
                                raw_out << local_raw;
                                local_raw.clear();
                            }
                        }
                    }
                }
            }
        }

        if (!local_raw.empty()) {
            std::lock_guard<std::mutex> lock(raw_mu);
            raw_out << local_raw;
        }
#pragma omp critical
        {
                Agg& dst = agg;
                const Agg& src = local_agg;
                dst.n += src.n;
                dst.n_near += src.n_near;
                dst.n_hard += src.n_hard;
                dst.n_far += src.n_far;
                dst.n_neg += src.n_neg;
                dst.n_prune += src.n_prune;
                dst.sum_rho += src.sum_rho;
                dst.sum_rho_hard += src.sum_rho_hard;
                dst.sum_lb2 += src.sum_lb2;
                dst.sum_d2 += src.sum_d2;
                dst.sum_gap_pos += src.sum_gap_pos;
                dst.sum_gap_neg += src.sum_gap_neg;
        }
    }

    raw_out.flush();

    std::ofstream sum_out(summary_path, std::ios::trunc);
    if (!sum_out) throw std::runtime_error("Cannot open lb probe summary: " + summary_path);
    sum_out << "scope,method,P,n,n_near,n_hard,n_far,mean_rho,mean_rho_hard,mean_lb2,mean_d2,"
               "neg_n,prune_rate_on_neg,mean_underprune_gap,mean_overprune_gap,"
               "nq,max_probes,max_per_list,triangle_survivors_only,boundary_alpha\n";
    {
        const Agg& a = agg;
        const double inv = a.n ? 1.0 / static_cast<double>(a.n) : 0.0;
        const double inv_h = a.n_hard ? 1.0 / static_cast<double>(a.n_hard) : 0.0;
        const double inv_neg = a.n_neg ? 1.0 / static_cast<double>(a.n_neg) : 0.0;
        const double prune_rate = a.n_neg ? static_cast<double>(a.n_prune) / a.n_neg : 0.0;
        sum_out << scope_name << ',' << multipivot_method << ',' << P << ',' << a.n << ','
                << a.n_near << ',' << a.n_hard << ',' << a.n_far << ',' << (a.sum_rho * inv) << ','
                << (a.sum_rho_hard * inv_h) << ',' << (a.sum_lb2 * inv) << ',' << (a.sum_d2 * inv)
                << ',' << a.n_neg << ',' << prune_rate << ','
                << (a.sum_gap_neg * inv_neg) << ',' << (a.sum_gap_pos * inv_neg) << ',' << n << ','
                << probes << ',' << (max_per_list == std::numeric_limits<size_t>::max() ? 0 : max_per_list)
                << ',' << (config.triangle_survivors_only ? 1 : 0) << ',' << alpha << '\n';
    }
    if (verbose) {
        std::cout << "lb_probe wrote raw=" << raw_csv_path << " summary=" << summary_path
                  << " raw_rows~=" << raw_rows.load() << " P=" << P << std::endl;
    }
}

void Index::probe_progressive_oracle(
    size_t n,
    const float* queries,
    size_t search_k,
    const std::string& csv_path,
    const ProgressiveOracleConfig& config) const {
    if (metric != MetricType::METRIC_L2 || multipivot_method != "pca") {
        throw std::invalid_argument("probe_progressive_oracle requires L2 PCA metadata");
    }
    if (multipivot_built_count < 1 || !lists || !centroid_codes || csv_path.empty()) {
        throw std::invalid_argument("probe_progressive_oracle: incomplete index or output path");
    }
    if (search_k == 0) {
        throw std::invalid_argument("probe_progressive_oracle: search_k must be positive");
    }

    const size_t P = multipivot_built_count;
    const size_t m = P - 1;
    const size_t probes = std::max<size_t>(
        1, std::min(config.max_probes == 0 ? nprobe : config.max_probes, nlist));
    std::vector<size_t> checkpoints = config.checkpoints;
    if (checkpoints.empty()) {
        checkpoints.push_back(0);
        if (m > 0) {
            for (size_t k = 1; k < m; k *= 2) checkpoints.push_back(k);
            checkpoints.push_back(m);
        }
    }
    for (size_t& k : checkpoints) k = std::min(k, m);
    checkpoints.push_back(0);
    checkpoints.push_back(m);
    std::sort(checkpoints.begin(), checkpoints.end());
    checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());

    prepareDirectory(csv_path);
    std::ofstream out(csv_path, std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open progressive oracle CSV: " + csv_path);
    out << "q_id,list_rank,list_id,list_size,tau2,scope,m,k,center_survivors,"
           "prefix_pruned,ub_exact,undecided,candidate_steps,residual_checks,exact_distances\n";
    std::mutex output_mu;

#pragma omp parallel
    {
        std::unique_ptr<IVFScanBase> scanner_quantizer =
            const_cast<Index*>(this)->get_scanner(metric, OPT_NONE, probes,
                                                   edge_device_enabled);
        std::vector<float> centroid_distances(probes);
        std::vector<idx_t> list_ids(probes);
        std::vector<float> exact_distances;
        std::vector<float> centered(d);
        std::array<double, MULTIPIVOT_MAX_PIVOTS> q_pivot_sq{};
        std::array<float, MULTIPIVOT_MAX_PIVOTS> q_z{};
        std::array<double, MULTIPIVOT_MAX_PIVOTS> global_q_pivot_sq{};
        std::array<float, MULTIPIVOT_MAX_PIVOTS> global_q_z{};
        std::array<uint16_t, MULTIPIVOT_MAX_PIVOTS> residual_prefix{};
        std::string local_output;
        local_output.reserve(1 << 18);

#pragma omp for schedule(dynamic, 1)
        for (size_t qi = 0; qi < n; ++qi) {
            const float* q = queries + qi * d;
            init_result(metric, probes, centroid_distances.data(), list_ids.data());
            scanner_quantizer->set_query(q);
            scanner_quantizer->lite_scan_codes(
                nlist, centroid_codes.get(),
                reinterpret_cast<const size_t*>(centroid_ids.get()),
                centroid_distances.data(), list_ids.data());
            sort_result(metric, probes, centroid_distances.data(), list_ids.data());

            // Oracle radius: final kth distance among exactly the lists selected by nprobe.
            size_t candidate_total = 0;
            for (size_t rank = 0; rank < probes; ++rank) {
                const idx_t lid = list_ids[rank];
                if (lid >= 0 && static_cast<size_t>(lid) < nlist) {
                    candidate_total += lists[lid].list_size;
                }
            }
            if (candidate_total < search_k) continue;
            exact_distances.clear();
            exact_distances.reserve(candidate_total);
            for (size_t rank = 0; rank < probes; ++rank) {
                const idx_t lid = list_ids[rank];
                if (lid < 0 || static_cast<size_t>(lid) >= nlist) continue;
                const IVF& list = lists[lid];
                for (size_t ci = 0; ci < list.list_size; ++ci) {
                    exact_distances.push_back(
                        calculatedEuclideanDistance(q, list.get_candidate_codes(ci), d));
                }
            }
            std::nth_element(exact_distances.begin(),
                             exact_distances.begin() + static_cast<std::ptrdiff_t>(search_k - 1),
                             exact_distances.end());
            const float tau2 = exact_distances[search_k - 1];
            if (!(tau2 >= 0.0f) || !std::isfinite(tau2)) continue;
            const float sqrt_tau = std::sqrt(tau2);

            float global_sqrt_rho = 0.0f;
            bool global_ready = false;
            if (multipivot_scope == MultiPivotScope::GLOBAL && global_pivots.count == P &&
                global_pivots.has_U(P)) {
                global_ready = computePcaQuerySignature(
                    q, d, global_pivots, P, global_q_pivot_sq.data(), global_q_z.data(),
                    &global_sqrt_rho, centered.data());
            }

            for (size_t rank = 0; rank < probes; ++rank) {
                const idx_t lid = list_ids[rank];
                if (lid < 0 || static_cast<size_t>(lid) >= nlist) continue;
                const IVF& list = lists[lid];
                const PivotMetadata& meta = list.pivots;
                if (list.list_size == 0 || meta.count != P || !meta.has_U(P) ||
                    meta.signature_precision != SignaturePrecision::FLOAT32 ||
                    meta.signature_data.size() != P * list.list_size ||
                    meta.candidate_centroid_squared.size() != list.list_size ||
                    list.get_sqrt_candidate2centroid() == nullptr) {
                    continue;
                }

                const float* zq = nullptr;
                const double* qp = nullptr;
                float sqrt_rho_q = 0.0f;
                bool ready = false;
                if (multipivot_scope == MultiPivotScope::GLOBAL) {
                    ready = global_ready;
                    zq = global_q_z.data();
                    qp = global_q_pivot_sq.data();
                    sqrt_rho_q = global_sqrt_rho;
                } else {
                    ready = computePcaQuerySignature(q, d, meta, P, q_pivot_sq.data(),
                                                     q_z.data(), &sqrt_rho_q,
                                                     centered.data());
                    zq = q_z.data();
                    qp = q_pivot_sq.data();
                }
                if (!ready || zq == nullptr || qp == nullptr) continue;

                std::vector<uint64_t> pruned(checkpoints.size(), 0);
                std::vector<uint64_t> exact_early(checkpoints.size(), 0);
                std::vector<uint64_t> steps(checkpoints.size(), 0);
                std::vector<uint64_t> residual_checks(checkpoints.size(), 0);
                uint64_t center_survivors = 0;
                const float sqrt_q_ivf =
                    std::sqrt(std::max(0.0f, centroid_distances[rank]));

                for (size_t ci = 0; ci < list.list_size; ++ci) {
                    const float rx_ivf = list.get_sqrt_candidate2centroid(ci);
                    const float center_gap = sqrt_q_ivf - rx_ivf;
                    if (center_gap * center_gap > tau2) continue;
                    ++center_survivors;

                    size_t event_stage = m + 1;
                    bool event_prune = false;
                    bool event_exact = false;
                    residual_prefix.fill(0);
                    uint16_t residual_count = 0;

                    double hq2 = std::max(0.0, qp[0]);
                    double hx2 = std::max(
                        0.0, static_cast<double>(meta.candidate_centroid_squared[ci]));
                    double hq = std::sqrt(hq2);
                    double hx = std::sqrt(hx2);
                    // In global mode this is an additional shared-center stage-0 bound.
                    const double lb0 = (hq - hx) * (hq - hx);
                    const double ub0 = (hq + hx) * (hq + hx);
                    if (lb0 > static_cast<double>(tau2)) {
                        event_stage = 0;
                        event_prune = true;
                    } else if (ub0 <= static_cast<double>(tau2)) {
                        event_stage = 0;
                        event_exact = true;
                    }

                    double projected = 0.0;
                    for (size_t dim = 1; dim <= m && event_stage > m; ++dim) {
                        const float qv = zq[dim - 1];
                        const float xv =
                            meta.signature_data[(dim - 1) * list.list_size + ci];
                        const double delta = static_cast<double>(qv) - xv;
                        projected += delta * delta;
                        hq2 = std::max(0.0, hq2 - static_cast<double>(qv) * qv);
                        hx2 = std::max(0.0, hx2 - static_cast<double>(xv) * xv);
                        if (projected > static_cast<double>(tau2)) {
                            event_stage = dim;
                            event_prune = true;
                            residual_prefix[dim] = residual_count;
                            break;
                        }
                        ++residual_count;
                        residual_prefix[dim] = residual_count;
                        hq = std::sqrt(hq2);
                        hx = std::sqrt(hx2);
                        const double gap = hq - hx;
                        if (projected + gap * gap > static_cast<double>(tau2)) {
                            event_stage = dim;
                            event_prune = true;
                            break;
                        }
                        const double sum = hq + hx;
                        if (projected + sum * sum <= static_cast<double>(tau2)) {
                            event_stage = dim;
                            event_exact = true;
                            break;
                        }
                    }
                    for (size_t ck = 0; ck < checkpoints.size(); ++ck) {
                        const size_t kdim = checkpoints[ck];
                        const size_t evaluated = std::min(kdim, event_stage);
                        steps[ck] += evaluated;
                        residual_checks[ck] += residual_prefix[evaluated];
                        if (event_prune && event_stage <= kdim) ++pruned[ck];
                        if (event_exact && event_stage <= kdim) ++exact_early[ck];
                    }
                }

                for (size_t ck = 0; ck < checkpoints.size(); ++ck) {
                    const uint64_t decided = pruned[ck] + exact_early[ck];
                    const uint64_t undecided =
                        center_survivors >= decided ? center_survivors - decided : 0;
                    const uint64_t exact_count = center_survivors - pruned[ck];
                    local_output += std::format(
                        "{},{},{},{},{:.9g},{},{},{},{},{},{},{},{},{},{}\n", qi,
                        rank, static_cast<long long>(lid), list.list_size, tau2,
                        multipivot_scope == MultiPivotScope::GLOBAL ? "global" : "per_list",
                        m, checkpoints[ck], center_survivors, pruned[ck],
                        exact_early[ck], undecided, steps[ck], residual_checks[ck],
                        exact_count);
                }
                if (local_output.size() >= (1u << 20)) {
                    std::lock_guard<std::mutex> lock(output_mu);
                    out << local_output;
                    local_output.clear();
                }
            }
        }

        if (!local_output.empty()) {
            std::lock_guard<std::mutex> lock(output_mu);
            out << local_output;
        }
    }
    out.flush();
    if (verbose) {
        std::cout << std::format(
            "progressive_oracle wrote {} queries={} probes={} P={} checkpoints={}\n",
            csv_path, n, probes, P, checkpoints.size());
    }
}

void Index::write_pivot_manifest(const std::string& path,
                                 const std::string& dataset,
                                 bool append) const {
    if (multipivot_built_count == 0 || path.empty()) return;
    prepareDirectory(path);
    const bool write_header = !append || !std::filesystem::exists(path) ||
                              std::filesystem::file_size(path) == 0;
    std::ofstream out(path, append ? std::ios::app : std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open pivot manifest " + path);
    if (write_header) {
        out << "dataset,index_format,pivot_format,scope,method,requested_P,actual_P,seed,list_id,"
               "source_ids,pairwise_distances,gram_eigenvalues,rank,condition,fallbacks,"
               "selection_seconds,candidate_prep_seconds,build_pivot_distance_computations\n";
    }
    auto write_row = [&](idx_t list_id, const PivotMetadata& metadata) {
        const size_t active = std::min(multipivot_built_count, metadata.count);
        out << dataset << ',' << INDEX_FORMAT_VERSION << ',' << MULTIPIVOT_FORMAT_VERSION << ','
            << multiPivotScopeName(multipivot_scope) << ',' << multipivot_method << ','
            << multipivot_built_count << ',' << active << ',' << multipivot_seed << ',' << list_id << ','
            << joinPivotIds(metadata, active) << ',' << joinPairwiseDistances(metadata, active) << ','
            << joinEigenvalues(metadata, active) << ',' << metadata.rank << ','
            << metadata.condition << ',' << metadata.fallback_count << ','
            << multipivot_selection_seconds << ',' << multipivot_candidate_distance_seconds << ','
            << multipivot_build_distance_computations << '\n';
    };
    if (multipivot_scope == MultiPivotScope::GLOBAL) {
        write_row(-1, global_pivots);
    } else {
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            write_row(static_cast<idx_t>(list_id), lists[list_id].pivots);
        }
    }
    std::cout << "pivot_manifest scope=" << multiPivotScopeName(multipivot_scope)
              << " method=" << multipivot_method
              << " P=" << multipivot_built_count
              << " seed=" << multipivot_seed
              << " lists=" << (multipivot_scope == MultiPivotScope::GLOBAL ? 1 : nlist)
              << " format=" << INDEX_FORMAT_VERSION << std::endl;

    if (!pca_cov_eigenvalues.empty()) {
        std::filesystem::path eig_path = std::filesystem::path(path).parent_path() /
                                         (std::filesystem::path(path).stem().string() + "_pca_cov_eigenvalues.csv");
        // Prefer sibling of manifest: replace filename.
        eig_path = std::filesystem::path(path);
        eig_path.replace_filename(eig_path.stem().string() + "_pca_cov_eigenvalues.csv");
        std::ofstream eig(eig_path);
        if (eig) {
            double total = 0;
            for (double v : pca_cov_eigenvalues) total += v;
            eig << "k,lambda,frac,cum_frac,lambda_over_lambda1,lambda_over_mean\n";
            double csum = 0;
            const double lam1 = pca_cov_eigenvalues.empty() ? 0.0 : pca_cov_eigenvalues[0];
            const double mean_lam =
                pca_cov_eigenvalues.empty() ? 0.0 : total / static_cast<double>(pca_cov_eigenvalues.size());
            for (size_t k = 0; k < pca_cov_eigenvalues.size(); ++k) {
                const double lam = pca_cov_eigenvalues[k];
                csum += lam;
                eig << (k + 1) << ',' << std::setprecision(17) << lam << ','
                    << (total > 0 ? lam / total : 0.0) << ',' << (total > 0 ? csum / total : 0.0) << ','
                    << (lam1 > 0 ? lam / lam1 : 0.0) << ',' << (mean_lam > 0 ? lam / mean_lam : 0.0)
                    << '\n';
            }
            std::cout << "pca_cov_eigenvalues written to " << eig_path << " (d=" << pca_cov_eigenvalues.size()
                      << " trace=" << total << ")" << std::endl;
        }
    }
}

void Index::load_SPANN(std::string path) {
    // std::filesystem::path p(path);
    // std::ifstream in(p / "selected.bin", std::ios::binary);
    // if (!in.is_open()) {
    //     throw std::runtime_error("Cannot open file " + (p / "selected.bin").string());
    // }
    // nlist = in.tellg() / sizeof(int32_t);
    // centroid_ids = std::make_unique<idx_t[]>(nlist);
    // in.seekg(0);
    // if (!in.read(reinterpret_cast<char*>(centroid_ids.get()), nlist * sizeof(int32_t))) {
    //     throw std::runtime_error("Cannot read file " + (p / "selected.bin").string());
    // }
    // in.close();

    // std::ifstream in2(p / "selection.bin", std::ios::binary);
    // size_t listno = 0;
    // while (true) {
    //     int32_t node, tonode;
    //     in2.read(reinterpret_cast<char*>(&node), sizeof(int32_t));
    //     in2.read(reinterpret_cast<char*>(&tonode), sizeof(int32_t));
    //     if (tonode == listno) {
    //         // lists[listno].load_SPANN(p, node);
    //     }
    //     if (in2.eof()) {
    //         break;
    //     }
    // }
}

}  // namespace tribase
