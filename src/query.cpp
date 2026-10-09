#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <omp.h>

#include <argparse/argparse.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "tribase.h"
#include "utils.h"

using namespace tribase;

bool str_lower_equal(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

struct ExternalGroundTruth {
    std::unique_ptr<idx_t[]> ids;
    size_t width = 0;
};

ExternalGroundTruth load_external_groundtruth(const std::string& path, size_t requested_nq) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open external ground truth: " + path);
    uint32_t stored_nq = 0, width = 0;
    const bool ivecs = path.ends_with(".ivecs");
    const bool i32bin = path.ends_with(".i32bin");
    if (!ivecs && !i32bin) {
        throw std::invalid_argument("External ground truth must be .ivecs or .i32bin: " + path);
    }
    if (i32bin) {
        input.read(reinterpret_cast<char*>(&stored_nq), sizeof(stored_nq));
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
    } else {
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
        if (!input || width == 0) throw std::runtime_error("Invalid ivecs ground truth: " + path);
        const uint64_t bytes = std::filesystem::file_size(path);
        const uint64_t stride = sizeof(uint32_t) * (static_cast<uint64_t>(width) + 1);
        if (bytes % stride != 0) throw std::runtime_error("Truncated ivecs ground truth: " + path);
        stored_nq = static_cast<uint32_t>(bytes / stride);
        input.seekg(0, std::ios::beg);
    }
    if (!input || width == 0 || requested_nq > stored_nq) {
        throw std::runtime_error("External ground truth shape is smaller than requested queries: " + path);
    }
    auto ids = std::make_unique<idx_t[]>(requested_nq * static_cast<size_t>(width));
    std::vector<int32_t> row(width);
    for (size_t query = 0; query < requested_nq; ++query) {
        if (ivecs) {
            uint32_t row_width = 0;
            input.read(reinterpret_cast<char*>(&row_width), sizeof(row_width));
            if (row_width != width) throw std::runtime_error("Mixed-width ivecs ground truth: " + path);
        }
        input.read(reinterpret_cast<char*>(row.data()),
                   static_cast<std::streamsize>(width * sizeof(int32_t)));
        if (!input) throw std::runtime_error("Truncated external ground truth: " + path);
        for (size_t neighbor = 0; neighbor < width; ++neighbor) {
            ids[query * width + neighbor] = static_cast<idx_t>(row[neighbor]);
        }
    }
    return {std::move(ids), static_cast<size_t>(width)};
}

float calculate_id_recall(const idx_t* labels, const idx_t* ground_truth,
                          size_t nq, size_t k, size_t ground_truth_width) {
    if (k > ground_truth_width) {
        throw std::invalid_argument("Search k exceeds external ground-truth width");
    }
    size_t hits = 0;
#pragma omp parallel for reduction(+ : hits)
    for (size_t query = 0; query < nq; ++query) {
        std::unordered_set<idx_t> expected(
            ground_truth + query * ground_truth_width,
            ground_truth + query * ground_truth_width + k);
        for (size_t neighbor = 0; neighbor < k; ++neighbor) {
            hits += expected.contains(labels[query * k + neighbor]);
        }
    }
    return static_cast<float>(hits) / static_cast<float>(nq * k);
}

bool is_byte_vector_file(const std::string& path) {
    return path.ends_with(".bvecs") || path.ends_with(".i8bin");
}

std::unique_ptr<float[]> load_byte_training_sample(const std::string& path,
                                                   size_t total,
                                                   size_t dimension,
                                                   size_t sample_count,
                                                   bool normalize) {
    if (sample_count == 0 || sample_count > total) {
        throw std::invalid_argument("Invalid streaming training sample size");
    }
    auto sample = std::make_unique<float[]>(sample_count * dimension);
    constexpr size_t sample_block_vectors = 8192;
    const size_t block_count =
        (sample_count + sample_block_vectors - 1) / sample_block_vectors;
    std::mt19937_64 random(6666);
    size_t output_begin = 0;
    size_t next_progress_block = std::max<size_t>(1, (block_count + 9) / 10);
    for (size_t block = 0; block < block_count; ++block) {
        const size_t rows =
            std::min(sample_block_vectors, sample_count - output_begin);
        const size_t stratum_begin = total * block / block_count;
        const size_t stratum_end = total * (block + 1) / block_count;
        const size_t latest_begin =
            stratum_end >= rows ? stratum_end - rows : stratum_begin;
        const size_t source_begin = latest_begin > stratum_begin
            ? std::uniform_int_distribution<size_t>(stratum_begin, latest_begin)(random)
            : stratum_begin;
        if (source_begin + rows > total || source_begin + rows >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Byte-vector range exceeds the current loader limit");
        }
        auto [batch, loaded, loaded_dimension] = loadXvecs(
            path, {static_cast<int>(source_begin + 1),
                   static_cast<int>(source_begin + rows)});
        if (loaded != rows || static_cast<size_t>(loaded_dimension) != dimension) {
            throw std::runtime_error("Unexpected training sample shape from " + path);
        }
        std::copy_n(batch.get(), rows * dimension,
                    sample.get() + output_begin * dimension);
        output_begin += rows;
        if (block + 1 >= next_progress_block || block + 1 == block_count) {
            std::cout << std::format(
                "Streaming byte index: training sample read {:.1f}% ({}/{})\n",
                100.0 * (block + 1) / block_count, output_begin, sample_count)
                      << std::flush;
            while (next_progress_block <= block + 1 &&
                   next_progress_block < block_count) {
                next_progress_block += std::max<size_t>(1, (block_count + 9) / 10);
            }
        }
    }
    if (normalize) {
        normalizeL2InPlace(sample.get(), sample_count, dimension);
    }
    return sample;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("tribase");
    program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("./benchmarks"));
    program.add_argument("--dataset").help("dataset name").default_value(std::string("HandOutlines"));
    program.add_argument("--input_format").help("format of the dataset").default_value(std::string("fvecs"));
    program.add_argument("--output_format").help("format of the output").default_value(std::string("bin"));
    program.add_argument("--groundtruth_path")
        .help("official ID ground truth (.ivecs or .i32bin); bypasses generated groundtruth_*.bin")
        .default_value(std::string(""));
    program.add_argument("--k")
        .help("number of nearest neighbors")
        .default_value(100ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--subk")
        .default_value(15ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nq")
        .default_value(0ul)
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobes")
        .default_value(std::vector<size_t>({0ul}))
        .nargs(0, 100)
        .help("number of clusters to search")
        .scan<'u', size_t>();
    program.add_argument("--opt_levels")
        .default_value(std::vector<std::string>({"OPT_NONE", "OPT_TRIANGLE", "OPT_SUBNN_L2", "OPT_SUBNN_IP",
                                                 "OPT_TRI_SUBNN_L2", "OPT_TRI_SUBNN_IP", "OPT_ALL"}))
        .nargs(0, 10)
        .help("optimization levels");
    program.add_argument("--multipivot_modes")
        .default_value(std::vector<std::string>({"none"}))
        .nargs(1, 3)
        .help("multi-pivot filters: none, max, projection");
    program.add_argument("--multipivot_scope")
        .default_value(std::string("global"))
        .help("pivot scope: global or per_list");
    program.add_argument("--multipivot_method")
        .default_value(std::string("affine_fps"))
        .help("pivot selection: affine_fps, pca, weighted_pca, irls_pca, or greedy_prune");
    program.add_argument("--irls_max_iter")
        .default_value(size_t{5})
        .help("IRLS iterations for --multipivot_method irls_pca")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--irls_residual_floor")
        .default_value(1e-6f)
        .help("IRLS residual floor ε for weight update w=α/max(h,ε)")
        .action([](const std::string& value) -> float { return std::stof(value); });
    program.add_argument("--greedy_pseudo_nq")
        .default_value(size_t{64})
        .help("greedy_prune: max pseudo-queries sampled from each pool")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--greedy_negatives_per_q")
        .default_value(size_t{64})
        .help("greedy_prune: max negatives per pseudo-query")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--greedy_power_iters")
        .default_value(size_t{8})
        .help("greedy_prune: power-iteration steps for proxy direction")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--greedy_proxy_candidates")
        .default_value(size_t{4})
        .help("greedy_prune: #proxy eigenvectors evaluated with exact G_t")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--greedy_boundary_alpha")
        .default_value(0.2f)
        .help("greedy_prune: prefer negatives with τ < ||q-x|| <= (1+α)τ")
        .action([](const std::string& value) -> float { return std::stof(value); });
    program.add_argument("--greedy_real_query_ratio")
        .default_value(0.0f)
        .help("greedy_prune: sample this fraction of the full query set for build-time training (0=pseudo only)")
        .action([](const std::string& value) -> float { return std::stof(value); });
    program.add_argument("--greedy_real_lists_per_query")
        .default_value(size_t{8})
        .help("greedy_prune+per_list: assign each training query to this many nearest lists")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--pivot_counts")
        .default_value(std::vector<size_t>({1ul}))
        .nargs(1)
        .help("pruning plan: 0=IVF-Flat, 1=centroid Triangle, 2..512=multi-pivot")
        .scan<'u', size_t>();
    program.add_argument("--active_pivot_count")
        .default_value(size_t{0})
        .help("search-time prefix of a larger PCA index; 0 uses all --pivot_counts")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--active_pivot_counts")
        .default_value(std::vector<size_t>{})
        .nargs(0, 512)
        .help("search-time PCA prefixes evaluated from one persisted Pmax index")
        .scan<'u', size_t>();
    program.add_argument("--pivot_seed")
        .default_value(uint64_t{0})
        .action([](const std::string& value) -> uint64_t { return std::stoull(value); });
    program.add_argument("--pivot_manifest")
        .default_value(std::string(""))
        .help("pivot manifest CSV path; defaults under the dataset result directory");
    program.add_argument("--signature_precision")
        .default_value(std::string("float32"))
        .help("candidate signature storage: float32, float16, or float8 (int8+scale)");
    program.add_argument("--projection_prefix_length")
        .default_value(size_t{0})
        .help("PCA projection LB: 0=legacy; N=one prefix-residual checkpoint after N coordinates")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--projection_block_size")
        .default_value(size_t{0})
        .help("fixed progressive PCA block size; 0=legacy full-signature filtering")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--projection_dynamic")
        .default_value(false)
        .implicit_value(true)
        .help("dynamic stage-major PCA filtering with active-candidate compaction");
    program.add_argument("--microblocks")
        .default_value(false)
        .implicit_value(true)
        .help("enable signature-space microblock AABB prune (list-internal clustering)");
    program.add_argument("--microblock_bytes")
        .default_value(size_t{16384})
        .help("target microblock payload size in bytes (default 16KiB)")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--train_only").default_value(false).implicit_value(true).help("train only");
    program.add_argument("--cache").default_value(false).implicit_value(true).help("use cached index");
    program.add_argument("--build_batch_vectors")
        .default_value(size_t{262144})
        .help("decoded vectors retained per streaming byte-index build batch")
        .action([](const std::string& value) -> size_t { return std::stoull(value); });
    program.add_argument("--coarse_builder")
        .default_value(std::string("tribase"))
        .help("index-build coarse assignment: tribase or ivf_hnsw");
    program.add_argument("--coarse_hnsw_m")
        .default_value(size_t{32})
        .action([](const std::string& value) -> size_t { return std::stoull(value); });
    program.add_argument("--coarse_hnsw_ef_construction")
        .default_value(size_t{200})
        .action([](const std::string& value) -> size_t { return std::stoull(value); });
    program.add_argument("--coarse_hnsw_ef_search")
        .default_value(size_t{128})
        .action([](const std::string& value) -> size_t { return std::stoull(value); });
    program.add_argument("--coarse_hnsw_query_batch")
        .default_value(size_t{8192})
        .help("maximum queries per Faiss HNSW coarse-assignment call")
        .action([](const std::string& value) -> size_t { return std::stoull(value); });
    program.add_argument("--centroids_path")
        .default_value(std::string(""))
        .help("shared IVF centroids: load if exists, otherwise save after train");
    program.add_argument("--from_index")
        .default_value(std::string(""))
        .help("upgrade pruning structures from an existing IVF index (shared clustering)");
    program.add_argument("--load_index")
        .default_value(std::string(""))
        .help("load one existing rich index read-only and select pruning only at search time");
    program.add_argument("--index_path")
        .default_value(std::string(""))
        .help("explicit output/cache index path; permits isolated shared-IVF experiment indexes");
    program.add_argument("--sub_nprobe_ratio")
        .default_value(1.0f)
        .help("ratio of the number of subNNs to the number of clusters")
        .action([](const std::string& value) -> float { return std::stof(value); });
    program.add_argument("--metric").default_value("l2").help("metric type");
    program.add_argument("--normalize")
        .default_value(false)
        .implicit_value(true)
        .help("L2-normalize base+query before train/search (unit sphere => cosine via L2)");
    program.add_argument("--run_faiss").default_value(false).implicit_value(true).help("run faiss");
    program.add_argument("--loop").default_value(1ul).action(
        [](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--warmup_loops").default_value(1ul)
        .help("untimed search passes before every configuration, including loop=1")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--benchmark_info").default_value(false).implicit_value(true)
        .help("print timing protocol and statistics build mode without loading data");
    program.add_argument("--nlist").default_value(0ul).action(
        [](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--verbose").default_value(false).implicit_value(true).help("verbose");
    program.add_argument("--ratios")
        .default_value(std::vector<float>({1.0f}))
        .nargs(0, 100)
        .help("search ratio")
        .scan<'f', float>();
    program.add_argument("--csv").help("csv result file").default_value(std::string(""));
    program.add_argument("--dump_list_stats")
        .help("dump per-partition (probed list) pruning stats to CSV; empty disables")
        .default_value(std::string(""));
    program.add_argument("--dataset_info")
        .help("only output dataset-info to csv file")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--early_stop").help("early stop").default_value(false).implicit_value(true);
    program.add_argument("--lb_probe")
        .help("offline LB quality probe CSV path (also writes <path>.summary.csv); empty disables")
        .default_value(std::string(""));
    program.add_argument("--lb_probe_only")
        .default_value(false)
        .implicit_value(true)
        .help("run --lb_probe then exit without timed search");
    program.add_argument("--lb_probe_max_probes")
        .default_value(size_t{32})
        .help("lb_probe: max IVF lists per query")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--lb_probe_max_per_list")
        .default_value(size_t{64})
        .help("lb_probe: stride-sample candidates per list (0=all)")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--lb_probe_max_raw_rows")
        .default_value(size_t{250000})
        .help("lb_probe: max raw CSV rows (0=unlimited)")
        .action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--lb_probe_all_candidates")
        .default_value(false)
        .implicit_value(true)
        .help("lb_probe: include triangle-prunable candidates (default: triangle survivors only)");
    program.add_argument("--lb_probe_boundary_alpha")
        .default_value(0.2f)
        .help("lb_probe: hard-band alpha for tau < d <= (1+α)τ")
        .action([](const std::string& value) -> float { return std::stof(value); });
    program.add_argument("--progressive_oracle")
        .help("write progressive PCA oracle query-list/checkpoint statistics to CSV")
        .default_value(std::string(""));
    program.add_argument("--progressive_oracle_only")
        .default_value(false)
        .implicit_value(true)
        .help("run --progressive_oracle then exit without timed search");
    program.add_argument("--oracle_checkpoints")
        .default_value(std::vector<size_t>{})
        .nargs(0, 32)
        .help("projected-dimension checkpoints; empty uses 0,1,2,4,...,m")
        .scan<'u', size_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (program.get<bool>("benchmark_info")) {
        std::cout << "benchmark_protocol=search_only_v2 stats_enabled="
#ifdef TRIBASE_ENABLE_STATS
                  << 1
#else
                  << 0
#endif
                  << std::endl;
        return 0;
    }

    std::vector<size_t> nprobes = program.get<std::vector<size_t>>("nprobes");
    std::vector<std::string> opt_levels_str = program.get<std::vector<std::string>>("opt_levels");
    std::vector<float> ratios = program.get<std::vector<float>>("ratios");
    std::vector<std::string> multipivot_mode_strings =
        program.get<std::vector<std::string>>("multipivot_modes");
    std::vector<size_t> pivot_counts = program.get<std::vector<size_t>>("pivot_counts");
    std::vector<MultiPivotMode> multipivot_modes;
    for (const std::string& value : multipivot_mode_strings) {
        if (str_lower_equal(value, "none")) multipivot_modes.push_back(MultiPivotMode::NONE);
        else if (str_lower_equal(value, "max")) multipivot_modes.push_back(MultiPivotMode::MAX);
        else if (str_lower_equal(value, "projection") || str_lower_equal(value, "proj"))
            multipivot_modes.push_back(MultiPivotMode::PROJECTION);
        else throw std::invalid_argument("Invalid --multipivot_modes value: " + value);
    }
    if (pivot_counts.size() != 1) {
        throw std::invalid_argument(
            "--pivot_counts accepts exactly one value; build separate indexes for different P");
    }
    const size_t requested_pivot_count = pivot_counts.front();
    if (requested_pivot_count > MULTIPIVOT_MAX_PIVOTS) {
        throw std::invalid_argument("--pivot_counts must be in 0..512");
    }

    // P is the public pruning-plan selector:
    //   P=0: plain IVF-Flat, with every pruning path disabled.
    //   P=1: the list centroid is the sole pivot, exactly the legacy Triangle path.
    //   P>=2: preserve the requested OptLevel and run MP after it, allowing
    //         Triangle + MP as a two-stage cascade.
    if (requested_pivot_count == 0) {
        opt_levels_str = {"OPT_NONE"};
        multipivot_modes = {MultiPivotMode::NONE};
    } else if (requested_pivot_count == 1) {
        opt_levels_str = {"OPT_TRIANGLE"};
        multipivot_modes = {MultiPivotMode::NONE};
    }
    const bool multipivot_enabled =
        std::any_of(multipivot_modes.begin(), multipivot_modes.end(),
                    [](MultiPivotMode mode) { return mode != MultiPivotMode::NONE; });
    const size_t built_pivot_count = multipivot_enabled ? requested_pivot_count : 0;
    const size_t requested_active_pivot_count =
        program.get<size_t>("active_pivot_count");
    std::vector<size_t> active_pivot_counts =
        program.get<std::vector<size_t>>("active_pivot_counts");
    if (requested_active_pivot_count != 0 && !active_pivot_counts.empty()) {
        throw std::invalid_argument(
            "use only one of --active_pivot_count and --active_pivot_counts");
    }
    if (active_pivot_counts.empty()) {
        active_pivot_counts.push_back(requested_active_pivot_count == 0
            ? built_pivot_count : requested_active_pivot_count);
    }
    for (const size_t active_pivot_count : active_pivot_counts) {
        if (multipivot_enabled &&
            (active_pivot_count < 2 || active_pivot_count > built_pivot_count)) {
            throw std::invalid_argument(
                "active pivot counts must be in 2..--pivot_counts for projection search");
        }
    }
    if (!multipivot_enabled &&
        (requested_active_pivot_count != 0 || active_pivot_counts.size() != 1 ||
         active_pivot_counts.front() != built_pivot_count)) {
        throw std::invalid_argument(
            "active pivot counts require a multipivot mode");
    }
    const bool active_prefix_sweep =
        !program.get<std::vector<size_t>>("active_pivot_counts").empty();
    const std::string multipivot_scope_string = program.get<std::string>("multipivot_scope");
    MultiPivotScope multipivot_scope;
    if (str_lower_equal(multipivot_scope_string, "global")) multipivot_scope = MultiPivotScope::GLOBAL;
    else if (str_lower_equal(multipivot_scope_string, "per_list") ||
             str_lower_equal(multipivot_scope_string, "per-list"))
        multipivot_scope = MultiPivotScope::PER_LIST;
    else throw std::invalid_argument("--multipivot_scope must be global or per_list");
    const std::string multipivot_method = program.get<std::string>("multipivot_method");
    if (!validMultipivotMethod(multipivot_method)) {
        throw std::invalid_argument(
            "--multipivot_method must be affine_fps, pca, weighted_pca, irls_pca, or greedy_prune");
    }
    const size_t irls_max_iter = program.get<size_t>("irls_max_iter");
    const float irls_residual_floor = program.get<float>("irls_residual_floor");
    if (multipivot_method == "irls_pca") {
        if (irls_max_iter == 0) {
            throw std::invalid_argument("--irls_max_iter must be >= 1");
        }
        if (!(irls_residual_floor > 0.0f) || !std::isfinite(irls_residual_floor)) {
            throw std::invalid_argument("--irls_residual_floor must be a positive finite float");
        }
    }
    GreedyPruneConfig greedy_config;
    greedy_config.pseudo_nq = program.get<size_t>("greedy_pseudo_nq");
    greedy_config.negatives_per_q = program.get<size_t>("greedy_negatives_per_q");
    greedy_config.power_iters = program.get<size_t>("greedy_power_iters");
    greedy_config.proxy_candidates = program.get<size_t>("greedy_proxy_candidates");
    greedy_config.boundary_alpha = program.get<float>("greedy_boundary_alpha");
    greedy_config.real_query_ratio = program.get<float>("greedy_real_query_ratio");
    greedy_config.real_lists_per_query = program.get<size_t>("greedy_real_lists_per_query");
    if (multipivot_method == "greedy_prune") {
        if (greedy_config.pseudo_nq == 0 || greedy_config.negatives_per_q == 0) {
            throw std::invalid_argument("--greedy_pseudo_nq/--greedy_negatives_per_q must be >= 1");
        }
        if (greedy_config.power_iters == 0 || greedy_config.proxy_candidates == 0) {
            throw std::invalid_argument(
                "--greedy_power_iters/--greedy_proxy_candidates must be >= 1");
        }
        if (greedy_config.real_query_ratio < 0.0f || greedy_config.real_query_ratio > 1.0f ||
            !std::isfinite(greedy_config.real_query_ratio)) {
            throw std::invalid_argument("--greedy_real_query_ratio must be in [0,1]");
        }
        if (greedy_config.real_query_ratio > 0.0f && greedy_config.real_lists_per_query == 0) {
            throw std::invalid_argument("--greedy_real_lists_per_query must be >= 1 when using real queries");
        }
    }
    const uint64_t pivot_seed = program.get<uint64_t>("pivot_seed");
    const std::string signature_precision_string = program.get<std::string>("signature_precision");
    if (!validSignaturePrecisionName(signature_precision_string)) {
        throw std::invalid_argument(
            "--signature_precision must be float32, float16, or float8");
    }
    const SignaturePrecision signature_precision = parseSignaturePrecision(signature_precision_string);
    const size_t projection_prefix_length = program.get<size_t>("projection_prefix_length");
    const size_t projection_block_size = program.get<size_t>("projection_block_size");
    const bool projection_dynamic = program.get<bool>("projection_dynamic");
    if (projection_prefix_length != 0 && signature_precision != SignaturePrecision::FLOAT32) {
        throw std::invalid_argument(
            "--projection_prefix_length currently requires --signature_precision float32");
    }
    const bool microblocks_enabled = program.get<bool>("microblocks");
    const size_t microblock_bytes = program.get<size_t>("microblock_bytes");
    if (microblocks_enabled && !multipivot_enabled) {
        throw std::invalid_argument("--microblocks requires a multipivot mode other than none");
    }
    if (microblock_bytes < sizeof(float)) {
        throw std::invalid_argument("--microblock_bytes too small");
    }

    size_t k = program.get<size_t>("k");
    size_t subk = program.get<size_t>("subk");
    size_t user_nq = program.get<size_t>("nq");

    std::vector<OptLevel> opt_levels;
    for (const auto& opt_level_str : opt_levels_str) {
        opt_levels.push_back(str2OptLevel(opt_level_str));
    }
    OptLevel added_opt_levels = OptLevel::OPT_NONE;

    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset = program.get<std::string>("dataset");
    std::string input_format = program.get<std::string>("input_format");
    std::string output_format = program.get<std::string>("output_format");
    std::string metric_str = program.get<std::string>("metric");
    bool run_faiss = program.get<bool>("run_faiss");
    MetricType metric;
    size_t loop = program.get<size_t>("loop");
    const size_t warmup_loops = program.get<size_t>("warmup_loops");
    if (loop == 0) throw std::invalid_argument("--loop must be positive");
    size_t nlist = program.get<size_t>("nlist");
    bool verbose = program.get<bool>("verbose");
    bool early_stop = program.get<bool>("early_stop");
    std::string dump_list_stats_path = program.get<std::string>("dump_list_stats");
    bool dump_list_stats = !dump_list_stats_path.empty();

    if (early_stop && (ratios[0] != 1 || ratios.size() != 1)) {
        throw std::invalid_argument("early_stop is only allowed when ratios is 1.0");
    }

    if (str_lower_equal(metric_str, "l2")) {
        metric = MetricType::METRIC_L2;
    } else if (str_lower_equal(metric_str, "ip")) {
        metric = MetricType::METRIC_IP;
    } else {
        throw std::runtime_error("Invalid metric type");
    }
    if (multipivot_enabled && metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("Multi-pivot filtering is supported only for --metric l2");
    }
    const bool normalize = program.get<bool>("normalize");
    if (normalize && metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("--normalize requires --metric l2 (cosine via unit-sphere L2)");
    }

    bool train_only = program.get<bool>("train_only");
    bool cache = program.get<bool>("cache");
    const size_t build_batch_vectors = program.get<size_t>("build_batch_vectors");
    if (build_batch_vectors == 0) {
        throw std::invalid_argument("--build_batch_vectors must be greater than zero");
    }
    const std::string coarse_builder = program.get<std::string>("coarse_builder");
    const bool coarse_ivf_hnsw = coarse_builder == "ivf_hnsw";
    if (!coarse_ivf_hnsw && coarse_builder != "tribase") {
        throw std::invalid_argument("--coarse_builder must be tribase or ivf_hnsw");
    }
    const size_t coarse_hnsw_m = program.get<size_t>("coarse_hnsw_m");
    const size_t coarse_hnsw_ef_construction =
        program.get<size_t>("coarse_hnsw_ef_construction");
    const size_t coarse_hnsw_ef_search = program.get<size_t>("coarse_hnsw_ef_search");
    const size_t coarse_hnsw_query_batch =
        program.get<size_t>("coarse_hnsw_query_batch");
    if (coarse_ivf_hnsw &&
        (coarse_hnsw_m == 0 || coarse_hnsw_ef_construction == 0 ||
         coarse_hnsw_ef_search == 0 || coarse_hnsw_query_batch == 0)) {
        throw std::invalid_argument("IVF+HNSW coarse parameters must be greater than zero");
    }
    std::string centroids_path = program.get<std::string>("centroids_path");
    std::string from_index_path = program.get<std::string>("from_index");
    std::string load_index_path = program.get<std::string>("load_index");
    if (!from_index_path.empty() && !load_index_path.empty()) {
        throw std::invalid_argument("use only one of --from_index and --load_index");
    }
    if (!from_index_path.empty() && !std::filesystem::exists(from_index_path)) {
        throw std::runtime_error("Shared source index does not exist: " + from_index_path);
    }
    float sub_nprobe_ratio = program.get<float>("sub_nprobe_ratio");

    std::string base_path = std::format("{}/{}/origin/{}_base.{}", benchmarks_path, dataset, dataset, input_format);
    std::string query_path = std::format("{}/{}/origin/{}_query.{}", benchmarks_path, dataset, dataset, input_format);
    // Separate GT / index when unit-normalized so unnormalized caches are not reused.
    const std::string unit_tag = normalize ? "_unit" : "";
    std::string groundtruth_path =
        std::format("{}/{}/result/groundtruth_{}{}.{}", benchmarks_path, dataset, k, unit_tag, output_format);
    const std::string external_groundtruth_path = program.get<std::string>("groundtruth_path");
    const bool external_groundtruth = !external_groundtruth_path.empty();
    std::string log_path;
    std::string tmp_csv_path = program.get<std::string>("csv");
    if (tmp_csv_path.length()) {
        log_path = tmp_csv_path;
    } else {
        if (!run_faiss) {
            log_path = std::format("{}/{}/result/log.csv", benchmarks_path, dataset);
        } else {
            log_path = std::format("{}/{}/result/log_faiss.csv", benchmarks_path, dataset);
        }
    }
    std::string pivot_manifest_path = program.get<std::string>("pivot_manifest");
    if (pivot_manifest_path.empty()) {
        pivot_manifest_path = std::format("{}/{}/result/pivot_manifest.csv", benchmarks_path, dataset);
    }

    // if (run_faiss && train_only) {
    //     throw std::invalid_argument("run_faiss && train_only is not allowed, run_faiss will not train at all.");
    // }

    size_t nb, d;
    std::unique_ptr<float[]> base = nullptr;
    std::tie(nb, d) = loadXvecsInfo(base_path);

    if (program.get<bool>("dataset_info")) {
        auto [nq, _] = loadXvecsInfo(query_path);
        std::ofstream ofs;
        ofs.open(tmp_csv_path.data());
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file: " << tmp_csv_path << std::endl;
            return 0;
        }
        ofs << "nb, nq, d" << std::endl;
        ofs << std::format("{}, {}, {}", nb, nq, d) << std::endl;
        return 0;
    }

    if (nlist == 0) {
        nlist = static_cast<size_t>(std::sqrt(nb));
    }
    if (nprobes.back() == 0) {
        nprobes.back() = nlist;
    }
    size_t sub_nlist = std::sqrt(nb / nlist);
    size_t sub_nprobe = std::max(static_cast<size_t>(sub_nlist * sub_nprobe_ratio), 1ul);
    if (verbose) {
        std::cout << std::format("sub_nlist: {} sub_nprobe: {}", sub_nlist, sub_nprobe) << std::endl;
    }

    for (const OptLevel& opt_level : opt_levels) {
        added_opt_levels = static_cast<OptLevel>(static_cast<int>(added_opt_levels) | static_cast<int>(opt_level));
    }
    if (verbose) {
        std::cout << std::format("Added optimization levels: {}", static_cast<int>(added_opt_levels)) << std::endl;
    }
    // nprobes.clear();
    // for (size_t val = 1; val <= nlist / 2; val *= 2) {
    //     nprobes.push_back(val);
    // }
    // nprobes.push_back(nlist);

    auto get_index_path = [&]() {
        int target = static_cast<int>(added_opt_levels);
        std::string method_tag;
        if (multipivot_method == "irls_pca") {
            method_tag = std::format("_irls{}_{:g}", irls_max_iter, irls_residual_floor);
        } else if (multipivot_method == "greedy_prune") {
            method_tag = std::format("_gq{}_nx{}_pi{}_pc{}", greedy_config.pseudo_nq,
                                     greedy_config.negatives_per_q, greedy_config.power_iters,
                                     greedy_config.proxy_candidates);
            if (greedy_config.real_query_ratio > 0.0f) {
                method_tag += std::format("_rqr{:g}_rl{}", greedy_config.real_query_ratio,
                                          greedy_config.real_lists_per_query);
            }
        }
        const std::string coarse_tag = coarse_ivf_hnsw
            ? std::format("_coarse_ivfhnsw_M{}_efc{}_efs{}", coarse_hnsw_m,
                          coarse_hnsw_ef_construction, coarse_hnsw_ef_search)
            : "";
        return std::format(
            "{}/{}/index/v{}_nlist_{}_metric_{}_opt_{}_subk_{}_subNprobeRatio_{}_mp_{}_{}_P{}_seed{}{}{}{}{}.index",
            benchmarks_path, dataset, INDEX_FORMAT_VERSION, nlist,
            metric == MetricType::METRIC_L2 ? "l2" : "ip",
            target, subk, sub_nprobe_ratio, multiPivotScopeName(multipivot_scope),
            multipivot_method, built_pivot_count, pivot_seed, method_tag,
            microblocks_enabled ? std::format("_mb{}", microblock_bytes) : "",
            coarse_tag, unit_tag);
    };

    auto get_faiss_index_path = [&]() {
        return std::format("{}/{}/index/faiss_index_nlist_{}{}.index", benchmarks_path, dataset, nlist, unit_tag);
    };

    std::string index_path = program.get<std::string>("index_path");
    if (index_path.empty()) index_path = get_index_path();
    std::string faiss_index_path = get_faiss_index_path();
    prepareDirectory(faiss_index_path);

    auto [query, nq, _] = loadXvecs(query_path);
    if (normalize) {
        normalizeL2InPlace(query.get(), nq, static_cast<size_t>(d));
        if (verbose) {
            std::cout << "Normalized query vectors to unit L2" << std::endl;
        }
    }
    const size_t full_query_count = nq;

    // Sample real training queries for greedy from the full query file (before --nq truncate).
    std::unique_ptr<float[]> greedy_train_codes;
    size_t greedy_train_count = 0;
    if (multipivot_method == "greedy_prune" && greedy_config.real_query_ratio > 0.0f) {
        greedy_train_count = std::max<size_t>(
            1, static_cast<size_t>(std::llround(static_cast<double>(full_query_count) *
                                                greedy_config.real_query_ratio)));
        greedy_train_count = std::min(greedy_train_count, full_query_count);
        std::vector<size_t> order(full_query_count);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 rng(pivot_seed == 0 ? 0xC0FFEEULL : pivot_seed);
        std::shuffle(order.begin(), order.end(), rng);
        greedy_train_codes = std::make_unique<float[]>(greedy_train_count * static_cast<size_t>(d));
        for (size_t i = 0; i < greedy_train_count; ++i) {
            std::memcpy(greedy_train_codes.get() + i * d,
                        query.get() + order[i] * d, sizeof(float) * static_cast<size_t>(d));
        }
        if (verbose) {
            std::cout << std::format(
                "greedy real-query sample: {}/{} ({:g}%) seed={}\n", greedy_train_count,
                full_query_count, 100.0 * greedy_config.real_query_ratio, pivot_seed);
        }
    }

    if (user_nq > 0 && user_nq < nq) {
        nq = user_nq;
    }
    if(verbose){
        std::cout << "nq: " << nq << std::endl;
    }

    auto bind_greedy_train = [&](Index& idx) {
        if (greedy_train_count > 0 && greedy_train_codes) {
            idx.set_greedy_train_queries(greedy_train_codes.get(), greedy_train_count);
        } else {
            idx.set_greedy_train_queries(nullptr, 0);
        }
    };

    size_t ground_truth_width = k;
    std::unique_ptr<idx_t[]> ground_truth_I;
    std::unique_ptr<float[]> ground_truth_D;
    if (external_groundtruth) {
        ExternalGroundTruth loaded = load_external_groundtruth(external_groundtruth_path, nq);
        ground_truth_width = loaded.width;
        ground_truth_I = std::move(loaded.ids);
    } else {
        ground_truth_I = std::make_unique<idx_t[]>(k * nq);
        ground_truth_D = std::make_unique<float[]>(k * nq);
    }

    std::string faiss_time_path =
        std::format("{}/{}/result/faiss_result_nlist_{}{}.txt", benchmarks_path, dataset, nlist, unit_tag);
    std::vector<double> faiss_time(nprobes.size(), 0.0);
    std::ifstream faiss_time_input(faiss_time_path);
    if (faiss_time_input.is_open()) {
        size_t nprobe;
        double time;
        float recall, r2;
        while (faiss_time_input >> nprobe >> time >> recall >> r2) {
            auto it = std::find(nprobes.begin(), nprobes.end(), nprobe);
            if (it != nprobes.end()) {
                faiss_time[std::distance(nprobes.begin(), it)] = time;
            }
        }
    } else {
        if (verbose) {
            std::cout << std::format("Faiss time file {} does not exist", faiss_time_path) << std::endl;
        }
    }

    faiss::IndexFlatL2 quantizer(d);
    std::unique_ptr<faiss::IndexIVFFlat> index_faiss = std::make_unique<faiss::IndexIVFFlat>(&quantizer, d, nlist);

    auto train_load_faiss = [&]() {
        if (!std::filesystem::exists(faiss_index_path)) {
            Stopwatch warch_faiss;
            if (verbose) {
                std::cout << std::format("Training Faiss index") << std::endl;
            }
            if (base == nullptr) {
                std::tie(base, nb, d) = loadXvecs(base_path);
                if (normalize) {
                    normalizeL2InPlace(base.get(), nb, static_cast<size_t>(d));
                    if (verbose) {
                        std::cout << "Normalized base vectors to unit L2" << std::endl;
                    }
                }
            }
            warch_faiss.reset();
            index_faiss->train(nb, base.get());
            if (verbose) {
                double faiss_train_elapsed = warch_faiss.elapsedSeconds(true);
                std::cout << std::format("train: {:.2f}s", faiss_train_elapsed) << std::endl;
                std::cout << std::format("Adding vectors to Faiss index") << std::endl;
            }
            index_faiss->add(nb, base.get());
            if (verbose) {
                double faiss_add_elapsed = warch_faiss.elapsedSeconds(true);
                std::cout << std::format("add: {:.2f}s", faiss_add_elapsed) << std::endl;
            }
            faiss::write_index(index_faiss.get(), faiss_index_path.c_str());
        } else {
            if (verbose) {
                std::cout << std::format("Load faiss index from {}", faiss_index_path) << std::endl;
            }
            index_faiss.reset(dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(faiss_index_path.c_str())));
        }
    };

    if (external_groundtruth) {
        if (verbose) {
            std::cout << std::format("Loaded official ground truth {} (width={})",
                                     external_groundtruth_path, ground_truth_width)
                      << std::endl;
        }
    } else if (!std::filesystem::exists(groundtruth_path)) {
        double faiss_groundtruth_time = 0.0;
        if (verbose) {
            std::cout << std::format("Groundtruth file {} does not exist", groundtruth_path) << std::endl;
        }

        train_load_faiss();

        if (verbose) {
            std::cout << std::format("Searching Faiss index") << std::endl;
        }
        index_faiss->nprobe = nlist;
        Stopwatch stopwatch;
        index_faiss->search(nq, query.get(), k, ground_truth_D.get(), ground_truth_I.get());
        faiss_groundtruth_time = stopwatch.elapsedSeconds();
        writeResultsToFile(ground_truth_I.get(), ground_truth_D.get(), nq, k, groundtruth_path);
        if (verbose) {
            std::cout << std::format("Groundtruth file {} created using {} s", groundtruth_path, faiss_groundtruth_time)
                      << std::endl;
        }
        if (nprobes.back() == nlist) {
            faiss_time.back() = faiss_groundtruth_time;
        }
    } else {
        if (verbose) {
            std::cout << std::format("Loading groundtruth file {}", groundtruth_path) << std::endl;
        }
        loadResults(groundtruth_path, ground_truth_I.get(), ground_truth_D.get(), nq, k);
        if (verbose) {
            std::cout << std::format("Groundtruth file loaded") << std::endl;
        }
    }

    if (run_faiss) {
        if (verbose) {
            std::cout << std::format("Running Faiss") << std::endl;
        }
        if (!index_faiss->is_trained) {
            train_load_faiss();
        }
        std::ofstream faiss_time_output(faiss_time_path);
        if (!faiss_time_output.is_open()) {
            std::cerr << std::format("Fail to open {}\n", faiss_time_path);
        } else {
            if (verbose) {
                std::cout << std::format("Output faiss time to {}\n", faiss_time_path);
            }
        }
        std::unique_ptr<float[]> tmp_faiss_dis = std::make_unique<float[]>(k * nq);
        std::unique_ptr<idx_t[]> tmp_faiss_labels = std::make_unique<idx_t[]>(k * nq);
        CsvWriter faiss_time_output_writer(log_path, {"dataset", "nlist", "nprobe", "time", "qps", "recall", "r2"},
                                           true, false);
        for (size_t i = 0; i < nprobes.size(); i++) {
            index_faiss->nprobe = nprobes[i];
            for (size_t warmup = 0; warmup < warmup_loops; ++warmup) {
                index_faiss->search(nq, query.get(), k, tmp_faiss_dis.get(), tmp_faiss_labels.get());
            }
            Stopwatch stopwatch;
            for (size_t j = 0; j < loop; j++) {
                index_faiss->search(nq, query.get(), k, tmp_faiss_dis.get(), tmp_faiss_labels.get());
            }
            faiss_time[i] = stopwatch.elapsedSeconds() / loop;
            float recall = external_groundtruth
                ? calculate_id_recall(tmp_faiss_labels.get(), ground_truth_I.get(), nq, k,
                                      ground_truth_width)
                : calculate_recall(tmp_faiss_labels.get(), tmp_faiss_dis.get(), ground_truth_I.get(),
                                   ground_truth_D.get(), nq, k, metric);
            float r2 = external_groundtruth
                ? std::numeric_limits<float>::quiet_NaN()
                : calculate_r2(tmp_faiss_labels.get(), tmp_faiss_dis.get(), ground_truth_I.get(),
                               ground_truth_D.get(), nq, k, metric);
            double qps = static_cast<double>(nq) / faiss_time[i];
            std::cout << std::format("Faiss nprobe: {} time: {} qps: {} recall: {} r2: {}", nprobes[i], faiss_time[i],
                                     qps, recall, r2)
                      << std::endl;
            faiss_time_output << std::format("{} {} {} {}", nprobes[i], faiss_time[i], recall, r2) << std::endl;
            faiss_time_output.flush();
            faiss_time_output_writer << dataset << nlist << nprobes[i] << faiss_time[i] << qps << recall << r2
                                     << std::endl;
        }
        return 0;
    }

    Index index;

    auto save_index_atomically = [&](const std::string& path) {
        const std::string partial_path = path + ".partial";
        index.save_index(partial_path);
        std::error_code error;
        std::filesystem::rename(partial_path, path, error);
        if (error) {
            throw std::runtime_error(
                "Cannot publish completed index " + path + ": " + error.message());
        }
    };

    auto apply_runtime_extras = [&]() {
        index.microblock_enabled = microblocks_enabled;
        index.microblock_bytes = microblock_bytes;
        index.signature_precision = signature_precision;
        index.apply_signature_precision();
        index.set_projection_prefix_length(projection_prefix_length);
        index.projection_block_size = projection_block_size;
        index.projection_dynamic = projection_dynamic;
        if (microblocks_enabled) {
            bool need_rebuild = false;
            for (size_t list_id = 0; list_id < index.nlist; ++list_id) {
                if (index.lists[list_id].list_size > 0 && index.lists[list_id].microblocks.empty()) {
                    need_rebuild = true;
                    break;
                }
            }
            if (need_rebuild) index.rebuild_microblocks();
        } else if (index.lists) {
            for (size_t list_id = 0; list_id < index.nlist; ++list_id) {
                index.lists[list_id].microblocks.clear();
            }
        }
    };

    if (!load_index_path.empty()) {
        if (!std::filesystem::exists(load_index_path)) {
            throw std::runtime_error("Read-only index does not exist: " + load_index_path);
        }
        if (verbose) {
            std::cout << std::format("Loading shared rich index from {}", load_index_path)
                      << std::endl;
        }
        index.load_index(load_index_path);
        const int requested_opt = static_cast<int>(added_opt_levels);
        const int persisted_opt = static_cast<int>(index.added_opt_level);
        if ((requested_opt & persisted_opt) != requested_opt) {
            throw std::runtime_error(
                "Shared index does not contain the requested pruning metadata");
        }
        if (multipivot_enabled &&
            (index.multipivot_scope != multipivot_scope ||
             index.multipivot_method != multipivot_method ||
             index.multipivot_seed != pivot_seed ||
             index.multipivot_built_count < built_pivot_count)) {
            throw std::runtime_error(
                "Shared index does not contain the requested multi-pivot geometry");
        }
        apply_runtime_extras();
        if (verbose) {
            std::cout << "Shared rich index loaded read-only" << std::endl;
        }
    } else if (std::filesystem::exists(index_path) && cache) {
        if (verbose) {
            std::cout << std::format("Loading index from {}", index_path) << std::endl;
        }
        index.load_index(index_path);
        if (index.multipivot_scope != multipivot_scope ||
            index.multipivot_method != multipivot_method ||
            index.multipivot_built_count != built_pivot_count ||
            index.multipivot_seed != pivot_seed) {
            throw std::runtime_error("Cached index multi-pivot metadata does not match CLI configuration");
        }
        // IRLS / greedy hyperparams are encoded in the index path; keep them on the live object.
        index.irls_max_iter = irls_max_iter;
        index.irls_residual_floor = irls_residual_floor;
        index.greedy_prune_config = greedy_config;
        bind_greedy_train(index);
        if (multipivot_enabled && multipivot_scope == MultiPivotScope::PER_LIST &&
            isCentroidAnchoredMultipivotMethod(multipivot_method) &&
            !(added_opt_levels & OptLevel::OPT_TRIANGLE) &&
            !(added_opt_levels & OptLevel::OPT_SUBNN_IP)) {
            if (verbose) {
                std::cout << "Rebuilding per-list centroid-anchored multipivot metadata to refresh radii"
                          << std::endl;
            }
            index.rebuild_multipivot_metadata();
        }
        apply_runtime_extras();
        if (verbose) {
            std::cout << std::format("Index loaded") << std::endl;
        }
    } else if (!from_index_path.empty() && std::filesystem::exists(from_index_path)) {
        // Shared clustering: load bare (or other) IVF and only materialize missing prune info.
        if (verbose) {
            std::cout << std::format("Upgrading pruning from {}", from_index_path) << std::endl;
        }
        index.load_index(from_index_path);
        if (index.d != d || index.nlist != nlist || index.metric != metric) {
            throw std::runtime_error("Shared source index dimensions/nlist/metric do not match this experiment");
        }
        index.sub_k = subk;
        index.sub_nlist = sub_nlist;
        index.sub_nprobe = sub_nprobe;
        index.verbose = verbose;
        if (multipivot_enabled) {
            index.configure_multipivot(multipivot_scope, multipivot_method, built_pivot_count, pivot_seed,
                                       irls_max_iter, irls_residual_floor, greedy_config);
            bind_greedy_train(index);
        } else {
            index.multipivot_scope = multipivot_scope;
            index.multipivot_built_count = 0;
            index.multipivot_active_count = 0;
            index.multipivot_method = multipivot_method;
            index.multipivot_seed = pivot_seed;
        }
        index.microblock_enabled = microblocks_enabled;
        index.microblock_bytes = microblock_bytes;
        index.signature_precision = SignaturePrecision::FLOAT32;
        index.upgrade_pruning(added_opt_levels);
        save_index_atomically(index_path);
        apply_runtime_extras();
        if (verbose) {
            std::cout << std::format("Upgraded index saved to {}", index_path) << std::endl;
        }
    } else {
        const bool streaming_byte_build = is_byte_vector_file(base_path);
        if (coarse_ivf_hnsw && !streaming_byte_build) {
            throw std::invalid_argument(
                "--coarse_builder ivf_hnsw currently requires bvecs or i8bin input");
        }
        index = Index(d, nlist, 0, metric, added_opt_levels, subk, sub_nlist, sub_nprobe, verbose);
        if (multipivot_enabled) {
            index.configure_multipivot(multipivot_scope, multipivot_method, built_pivot_count, pivot_seed,
                                       irls_max_iter, irls_residual_floor, greedy_config);
            bind_greedy_train(index);
        } else {
            // Persist the requested identity fields even without MP metadata so
            // the generated cache path can be loaded and validated later.
            index.multipivot_scope = multipivot_scope;
            index.multipivot_method = multipivot_method;
            index.multipivot_seed = pivot_seed;
        }
        index.microblock_enabled = microblocks_enabled;
        index.microblock_bytes = microblock_bytes;
        // Build always materializes float32 signatures.
        index.signature_precision = SignaturePrecision::FLOAT32;

        auto build_start = std::chrono::high_resolution_clock::now();
        const bool have_centroids =
            !centroids_path.empty() && std::filesystem::exists(centroids_path);
        if (have_centroids) {
            index.load_centroids(centroids_path);
        } else {
            if (streaming_byte_build) {
                constexpr size_t max_points_per_centroid = 256;
                const size_t max_training_vectors =
                    nlist > std::numeric_limits<size_t>::max() / max_points_per_centroid
                    ? nb
                    : nlist * max_points_per_centroid;
                const size_t training_vectors = std::min(nb, max_training_vectors);
                std::cout << std::format(
                    "Streaming byte index: training sample={} of {}, add batch={} "
                    "centroid_backend={}\n",
                    training_vectors, nb, build_batch_vectors,
                    coarse_ivf_hnsw ? "ivf_hnsw" : "tribase") << std::flush;
                auto training_codes = load_byte_training_sample(
                    base_path, nb, d, training_vectors, normalize);
                std::cout << "Streaming byte index: centroid training started" << std::endl;
                if (coarse_ivf_hnsw) {
                    index.train_ivf_hnsw(
                        training_vectors, training_codes.get(), coarse_hnsw_m,
                        coarse_hnsw_ef_construction, coarse_hnsw_ef_search, 20, 6666);
                } else {
                    index.train(training_vectors, training_codes.get());
                }
                std::cout << "Streaming byte index: centroid training completed" << std::endl;
            } else {
                std::tie(base, nb, d) = loadXvecs(base_path);
                if (normalize) {
                    normalizeL2InPlace(base.get(), nb, static_cast<size_t>(d));
                    if (verbose) {
                        std::cout << "Normalized base vectors to unit L2" << std::endl;
                    }
                }
                index.train(nb, base.get());
            }
            if (!centroids_path.empty()) {
                index.save_centroids(centroids_path);
            }
        }
        if (streaming_byte_build) {
            auto loader = [&](size_t begin, size_t count) {
                if (begin + count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error(
                        "Byte-vector batch exceeds the current loader range");
                }
                auto [batch, loaded, loaded_dimension] = loadXvecs(
                    base_path,
                    {static_cast<int>(begin + 1), static_cast<int>(begin + count)});
                if (loaded != count || static_cast<size_t>(loaded_dimension) != d) {
                    throw std::runtime_error("Unexpected streaming batch shape from " + base_path);
                }
                if (normalize) {
                    normalizeL2InPlace(batch.get(), count, d);
                }
                return batch;
            };
            Index::HnswAssignmentConfig hnsw_assignment;
            hnsw_assignment.enabled = coarse_ivf_hnsw;
            hnsw_assignment.m = coarse_hnsw_m;
            hnsw_assignment.ef_construction = coarse_hnsw_ef_construction;
            hnsw_assignment.ef_search = coarse_hnsw_ef_search;
            hnsw_assignment.query_batch_size = coarse_hnsw_query_batch;
            index.add_batched(nb, build_batch_vectors, loader, hnsw_assignment);
        } else {
            if (!base) {
                std::tie(base, nb, d) = loadXvecs(base_path);
                if (normalize) {
                    normalizeL2InPlace(base.get(), nb, static_cast<size_t>(d));
                }
            }
            index.add(nb, base.get());
            base.reset();
        }
        auto build_end = std::chrono::high_resolution_clock::now();
        if (verbose) {
            std::cout << std::format("Index trained") << std::endl;
            std::cout << std::format("Index elapsed: {}s",
                                     std::chrono::duration<double>(build_end - build_start).count())
                      << std::endl;
        }
        save_index_atomically(index_path);
        apply_runtime_extras();
        if (verbose) {
            std::cout << std::format("Index saved to {}", index_path) << std::endl;
        }
    }
    if (index.d != d || index.nlist != nlist || index.metric != metric) {
        throw std::runtime_error("Loaded index dimensions/nlist/metric do not match this experiment");
    }
    if (multipivot_enabled) {
        index.write_pivot_manifest(pivot_manifest_path, dataset, true);
    }

    if (train_only) {
        return 0;
    }

    const std::string lb_probe_path = program.get<std::string>("lb_probe");
    const bool lb_probe_only = program.get<bool>("lb_probe_only");
    if (!lb_probe_path.empty()) {
        if (external_groundtruth) {
            throw std::invalid_argument(
                "--lb_probe requires ground-truth distances; official ID-only ground truth is insufficient");
        }
        if (!multipivot_enabled) {
            throw std::invalid_argument("--lb_probe requires a multipivot mode other than none");
        }
        if (metric != MetricType::METRIC_L2) {
            throw std::invalid_argument("--lb_probe requires --metric l2");
        }
        // Ensure nprobe is set for list selection (first requested nprobe).
        if (!nprobes.empty()) index.nprobe = nprobes.front();
        index.set_multipivot_search(MultiPivotMode::PROJECTION, built_pivot_count);
        Index::LbProbeConfig lb_cfg;
        lb_cfg.max_probes = program.get<size_t>("lb_probe_max_probes");
        lb_cfg.max_candidates_per_list = program.get<size_t>("lb_probe_max_per_list");
        lb_cfg.max_raw_rows = program.get<size_t>("lb_probe_max_raw_rows");
        lb_cfg.triangle_survivors_only = !program.get<bool>("lb_probe_all_candidates");
        lb_cfg.boundary_alpha = program.get<float>("lb_probe_boundary_alpha");
        // Oracle tau^2 = GT k-NN distance (Faiss L2 distances are squared).
        std::unique_ptr<float[]> tau2 = std::make_unique<float[]>(nq);
        for (size_t qi = 0; qi < nq; ++qi) {
            tau2[qi] = ground_truth_D[qi * k + (k - 1)];
        }
        if (verbose) {
            std::cout << std::format(
                "lb_probe: path={} nq={} P={} probes={} per_list={} tri_only={} alpha={}\n",
                lb_probe_path, nq, built_pivot_count, lb_cfg.max_probes,
                lb_cfg.max_candidates_per_list, lb_cfg.triangle_survivors_only, lb_cfg.boundary_alpha);
        }
        index.probe_lower_bounds(nq, query.get(), tau2.get(), lb_probe_path, lb_cfg);
        if (lb_probe_only) {
            return 0;
        }
    }

    const std::string progressive_oracle_path =
        program.get<std::string>("progressive_oracle");
    if (!progressive_oracle_path.empty()) {
        if (!multipivot_enabled || multipivot_method != "pca" ||
            multipivot_modes.end() == std::find(multipivot_modes.begin(), multipivot_modes.end(),
                                                MultiPivotMode::PROJECTION)) {
            throw std::invalid_argument(
                "--progressive_oracle requires PCA projection multipivot mode");
        }
        if (metric != MetricType::METRIC_L2) {
            throw std::invalid_argument("--progressive_oracle requires --metric l2");
        }
        if (signature_precision != SignaturePrecision::FLOAT32) {
            throw std::invalid_argument(
                "--progressive_oracle currently requires --signature_precision float32");
        }
        if (!nprobes.empty()) index.nprobe = nprobes.front();
        index.set_multipivot_search(MultiPivotMode::PROJECTION, built_pivot_count);
        Index::ProgressiveOracleConfig oracle_cfg;
        oracle_cfg.max_probes = nprobes.empty() ? 0 : nprobes.front();
        oracle_cfg.checkpoints = program.get<std::vector<size_t>>("oracle_checkpoints");
        index.probe_progressive_oracle(nq, query.get(), k, progressive_oracle_path,
                                       oracle_cfg);
        if (program.get<bool>("progressive_oracle_only")) return 0;
    }

    if (std::getenv("EDGE_DEVICE_ENABLED") != nullptr) {
        index.edge_device_enabled = EdgeDevice::EDGEDEVIVE_ENABLED;
        std::cout << "Edge device enabled" << std::endl;
    }

    if (dump_list_stats) {
        index.collect_list_stats = true;
        if (verbose) {
            std::cout << std::format("Dumping per-list stats to {}", dump_list_stats_path) << std::endl;
        }
        // Keep one measured pass when dumping. Warmup statistics are discarded,
        // so only the final measured pass supplies list_visits.
        if (loop > 1) {
            std::cerr << "Warning: --dump_list_stats forces loop=1 (was " << loop << ")" << std::endl;
            loop = 1;
        }
    }

    bool list_stats_header_written = false;
    std::vector<std::pair<MultiPivotMode, size_t>> multipivot_runs;
    for (MultiPivotMode mode : multipivot_modes) {
        if (mode == MultiPivotMode::NONE) {
            // Keep P=0/P=1 visible in result names and CSVs even though neither
            // materializes multi-pivot metadata.
            multipivot_runs.emplace_back(
                mode, requested_pivot_count <= 1 ? requested_pivot_count : 0);
        } else {
            for (const size_t active_pivot_count : active_pivot_counts) {
                multipivot_runs.emplace_back(mode, active_pivot_count);
            }
        }
    }
    for (size_t i = 0; i < nprobes.size(); i++) {
        size_t nprobe = nprobes[i];
        double f_time = faiss_time[i];
        index.nprobe = nprobe;
        bool early_stop_flag = false;
        for (const OptLevel& opt_level : opt_levels) {
            index.opt_level = opt_level;
            for (const auto& [multipivot_mode, pivot_count] : multipivot_runs) {
                const size_t runtime_prefix_length =
                    active_prefix_sweep && multipivot_mode == MultiPivotMode::PROJECTION
                    ? pivot_count - 1 : projection_prefix_length;
                if (active_prefix_sweep && multipivot_mode == MultiPivotMode::PROJECTION) {
                    // Materialize the candidate residual for this static prefix
                    // outside the timed query region.
                    index.set_projection_prefix_length(runtime_prefix_length);
                }
                index.set_multipivot_search(multipivot_mode, pivot_count);
            for (float ratio : ratios) {
                std::string output_path =
                    std::format("{}/{}/result/result_nlist_{}_nprobe_{}_opt_{}_mp_{}_P{}_k_{}_ratio_{}.{}",
                                benchmarks_path, dataset, nlist, nprobe, static_cast<int>(opt_level),
                                multiPivotModeName(multipivot_mode), pivot_count, k, ratio, output_format);
                std::unique_ptr<float[]> distances = std::make_unique<float[]>(nq * k);
                std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(nq * k);
                Stopwatch warmup_stopwatch;
                for (size_t warmup = 0; warmup < warmup_loops; ++warmup) {
                    index.search(nq, query.get(), k, distances.get(), labels.get(), ratio);
                }
                const double warmup_seconds = warmup_stopwatch.elapsedSeconds();
                Stopwatch stopwatch;
                Stats stats;
                for (size_t j = 0; j < loop; j++) {
                    stats = index.search(nq, query.get(), k, distances.get(), labels.get(), ratio);
                }
                // Freeze the search timer before recall/r2 launch their own work.
                const double search_time = stopwatch.elapsedSeconds() / loop;
                Stopwatch evaluation_stopwatch;
                float recall = external_groundtruth
                    ? calculate_id_recall(labels.get(), ground_truth_I.get(), nq, k, ground_truth_width)
                    : calculate_recall(labels.get(), distances.get(), ground_truth_I.get(),
                                       ground_truth_D.get(), nq, k, metric);
                float r2 = external_groundtruth
                    ? std::numeric_limits<float>::quiet_NaN()
                    : calculate_r2(labels.get(), distances.get(), ground_truth_I.get(),
                                   ground_truth_D.get(), nq, k, metric);
                stats.benchmark_protocol = "search_only_v2";
                stats.measurement_loops = loop;
                stats.warmup_loops = warmup_loops;
                stats.warmup_seconds = warmup_seconds;
                stats.evaluation_seconds = evaluation_stopwatch.elapsedSeconds();
                stats.n_query = nq;
                stats.simi_ratio = ratio;
                stats.nlist = nlist;
                stats.nprobe = nprobe;
                stats.query_time = search_time;
                stats.faiss_query_time = f_time;
                stats.opt_level = opt_level;
                stats.multipivot_mode = multipivot_mode;
                stats.multipivot_scope = multipivot_scope;
                stats.multipivot_method = multipivot_method;
                stats.pivot_count = pivot_count;
                stats.pivot_seed = pivot_seed;
                stats.signature_precision = signaturePrecisionName(signature_precision);
                stats.projection_prefix_length = runtime_prefix_length;
                stats.recall = recall;
                stats.r2 = r2;
                stats.print();
                stats.toCsv(log_path, true, dataset);
                if (dump_list_stats && !stats.list_visits.empty()) {
                    prepareDirectory(dump_list_stats_path);
                    listVisitsToCsv(dump_list_stats_path, stats.list_visits, list_stats_header_written, dataset,
                                    nlist, nprobe, opt_level, multipivot_mode, multipivot_scope,
                                    multipivot_method, pivot_count, pivot_seed, ratio);
                    list_stats_header_written = true;
                    if (verbose) {
                        std::cout << std::format("Wrote {} list visits to {}", stats.list_visits.size(),
                                                 dump_list_stats_path)
                                  << std::endl;
                    }
                }
                if (recall == 1) {
                    early_stop_flag = true;
                }
            }
            }
        }
        if (early_stop_flag && early_stop) {
            break;
        }
    }
}
