#ifndef INDEX_H
#define INDEX_H

#include <functional>
#include <memory>
#include <string>
#include "Clustering.h"
#include "IVF.h"
#include "IVFScan.hpp"
#include "common.h"

namespace tribase {

class Index {
   public:
    Index(size_t d = 0,
          size_t nlist = 0,
          size_t nprobe = 0,
          MetricType metric = MetricType::METRIC_L2,
          OptLevel opt_level = OptLevel::OPT_NONE,
          size_t sub_k = 0,
          size_t sub_nlist = 1,
          size_t sub_nprobe = 1,
          bool verbose = false,
          EdgeDevice edge_device_enabled = EdgeDevice::EDGEDEVIVE_DISABLED);

    Index& operator=(Index&& other) noexcept;

    void train(size_t n, const float* codes, bool faiss = false, bool lite = false);
    // Large-scale IVF coarse clustering. HNSW is used only as the assignment
    // index inside Lloyd iterations; the resulting index remains ordinary IVF.
    void train_ivf_hnsw(size_t n,
                        const float* codes,
                        size_t hnsw_m,
                        size_t ef_construction,
                        size_t ef_search,
                        size_t niter,
                        int seed);
    // Persist / restore IVF centroids only (shared clustering across pruning configs).
    void save_centroids(const std::string& path) const;
    void load_centroids(const std::string& path);

    void single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels);
    void single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio, Stats* stats);

    void add(size_t n, const float* codes);
    // Build IVF lists without materializing the complete input matrix. The
    // loader is invoked twice (counting and filling passes) with zero-based
    // [begin, begin + count) ranges and must return count*d floats.
    using BatchLoader =
        std::function<std::unique_ptr<float[]>(size_t begin, size_t count)>;
    struct HnswAssignmentConfig {
        bool enabled = false;
        size_t m = 32;
        size_t ef_construction = 200;
        size_t ef_search = 128;
        // Faiss HNSW search has per-call scratch state. Keep the coarse
        // assignment call bounded even when the decoded I/O batch is large.
        size_t query_batch_size = 8192;
    };
    void add_batched(size_t n,
                     size_t batch_size,
                     const BatchLoader& loader,
                     HnswAssignmentConfig hnsw);
    // Fill triangle radii from list codes + centroids (no re-assign).
    void ensure_triangle_radii();
    // Build OPT_SUBNN_* graphs for current opt_level bits (lists must already hold codes).
    void build_subnn_structures();
    // Upgrade a loaded bare IVF to `target` opt bits + configured multipivot metadata.
    void upgrade_pruning(OptLevel target);

    Stats search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio = 1.0);
    void save_index(std::string path) const;
    void load_index(std::string path);
    // Load vanilla (pre-TRIBASE2) index: IVF lists without pivot metadata.
    void load_legacy_index(std::string path);
    void load_SPANN(std::string path);
    void configure_multipivot(MultiPivotScope scope,
                              const std::string& method,
                              size_t pivot_count,
                              uint64_t seed,
                              size_t irls_max_iter = 5,
                              float irls_residual_floor = 1e-6f,
                              GreedyPruneConfig greedy_config = {});
    // Build/rebuild pivot metadata on an already-loaded IVF (no retrain/re-add).
    void rebuild_multipivot_metadata();
    void apply_signature_precision();
    void set_projection_prefix_length(size_t prefix_length);
    void rebuild_microblocks();
    void set_multipivot_search(MultiPivotMode mode, size_t pivot_count);
    void write_pivot_manifest(const std::string& path,
                              const std::string& dataset,
                              bool append) const;

    // Provide real queries for greedy_prune build (copied). Cleared if n==0.
    void set_greedy_train_queries(const float* queries, size_t n);

    // Offline LB quality probe for the single persisted P.
    // Does not mutate pivots. Writes `path` (raw sample) and `path` with ".summary.csv".
    struct LbProbeConfig {
        size_t max_probes = 32;              // max probed IVF lists per query
        size_t max_candidates_per_list = 64;  // stride-sample within each list
        size_t max_raw_rows = 250000;         // cap raw CSV rows (0 = unlimited)
        bool triangle_survivors_only = true; // drop candidates triangle would prune vs tau2
        float boundary_alpha = 0.2f;         // hard band: tau < d <= (1+α)τ
        bool append_raw = false;
    };
    // tau2_per_query: squared L2 of current top-k threshold (e.g. GT kNN distance).
    // Emits one row per sampled pair at multipivot_built_count.
    void probe_lower_bounds(size_t n,
                            const float* queries,
                            const float* tau2_per_query,
                            const std::string& raw_csv_path,
                            const LbProbeConfig& config) const;

    // Offline progressive-PCA oracle. For each query/list and checkpoint k, this
    // records the work remaining when the final top-k radius inside the probed
    // IVF lists is known in advance. It does not alter the online search path.
    struct ProgressiveOracleConfig {
        size_t max_probes = 0;  // 0 => current nprobe
        std::vector<size_t> checkpoints;  // projected dimensions; empty => powers of two + max
    };
    void probe_progressive_oracle(size_t n,
                                  const float* queries,
                                  size_t search_k,
                                  const std::string& csv_path,
                                  const ProgressiveOracleConfig& config) const;

    // 其他查询方法的声明

   private:
    std::unique_ptr<IVFScanBase> get_scanner(MetricType metric, OptLevel opt_level, size_t k, EdgeDevice edge_device_enabled = EdgeDevice::EDGEDEVIVE_DISABLED);

   public:
    size_t d;
    size_t nlist;
    size_t nprobe;
    MetricType metric;
    OptLevel opt_level;
    OptLevel added_opt_level;

    size_t sub_k;
    size_t sub_nlist;
    size_t sub_nprobe;

    bool verbose;
    EdgeDevice edge_device_enabled;
    // When true, search() records per-list visit stats into Stats::list_visits.
    bool collect_list_stats = false;

    MultiPivotScope multipivot_scope = MultiPivotScope::GLOBAL;
    MultiPivotMode multipivot_mode = MultiPivotMode::NONE;
    std::string multipivot_method = "affine_fps";
    size_t multipivot_built_count = 0;
    size_t multipivot_active_count = 0;
    // 0 keeps the legacy full-signature lower bound. A positive value adds one
    // prefix-residual checkpoint after this many projected coordinates.
    size_t projection_prefix_length = 0;
    // Optional fixed-size progressive PCA blocks; 0 keeps the legacy full scan.
    size_t projection_block_size = 0;
    bool projection_dynamic = false;
    uint64_t multipivot_seed = 0;
    size_t irls_max_iter = 5;
    float irls_residual_floor = 1e-6f;
    GreedyPruneConfig greedy_prune_config{};
    // Optional build-time real queries for greedy_prune (not persisted in index file).
    std::vector<float> greedy_train_queries;  // n * d, row-major
    size_t greedy_train_nq = 0;

    PivotMetadata global_pivots;
    double multipivot_selection_seconds = 0;
    double multipivot_candidate_distance_seconds = 0;
    uint64_t multipivot_build_distance_computations = 0;
    // Descending covariance eigenvalues from last PCA-family selection.
    std::vector<double> pca_cov_eigenvalues;
    std::vector<double> irls_objective_history;
    std::vector<double> greedy_gain_history;

    SignaturePrecision signature_precision = SignaturePrecision::FLOAT32;

    // Microblock AABB prune (signature-space); no disk I/O simulation.
    bool microblock_enabled = false;
    size_t microblock_bytes = MICROBLOCK_DEFAULT_BYTES;
    double microblock_build_seconds = 0;

    std::unique_ptr<IVF[]> lists;
    std::unique_ptr<float[]> centroid_codes;
    std::unique_ptr<idx_t[]> centroid_ids;
};

}  // namespace tribase

#endif  // INDEX_H
