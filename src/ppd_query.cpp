#include <argparse/argparse.hpp>
#include <omp.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Index.h"
#include "ExternalGroundTruth.h"
#include "Ppd.h"
#include "utils.h"

using namespace tribase;

int main(int argc, char** argv) {
    argparse::ArgumentParser program("ppd_query");
    program.add_argument("--index").required();
    program.add_argument("--query").required();
    program.add_argument("--groundtruth").default_value(std::string{});
    program.add_argument("--sidecar").required();
    program.add_argument("--nprobes").nargs(argparse::nargs_pattern::at_least_one)
        .scan<'u', size_t>().required();
    program.add_argument("--k").scan<'u', size_t>().default_value(size_t{1});
    program.add_argument("--nq").scan<'u', size_t>().default_value(size_t{0});
    program.add_argument("--block-size").scan<'u', size_t>().default_value(size_t{16});
    program.add_argument("--train-samples").scan<'u', size_t>().default_value(size_t{65536});
    program.add_argument("--seed").scan<'u', uint64_t>().default_value(uint64_t{6666});
    program.add_argument("--loop").scan<'u', size_t>().default_value(size_t{1});
    program.add_argument("--warmup-loops").scan<'u', size_t>().default_value(size_t{1});
    program.add_argument("--normalize").flag().help("normalize queries before timing; requires a unit-vector IVF index");
    program.add_argument("--build-only").flag().help("prepare/load sidecar and exit before search");
    program.add_argument("--no-save-sidecar").flag().help(
        "build PPD in memory and run queries without writing the transformed database");
    program.add_argument("--dump-search-results").default_value(std::string{});
    program.add_argument("--triangle").flag();
    program.add_argument("--rebuild-sidecar").flag();
    program.add_argument("--csv").default_value(std::string{});
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n' << program;
        return 2;
    }

    const std::string index_path = program.get<std::string>("--index");
    const std::string query_path = program.get<std::string>("--query");
    const std::string gt_path = program.get<std::string>("--groundtruth");
    const std::string sidecar_path = program.get<std::string>("--sidecar");
    const std::string csv_path = program.get<std::string>("--csv");
    const size_t k = program.get<size_t>("--k");
    const size_t block_size = program.get<size_t>("--block-size");
    const size_t train_samples = program.get<size_t>("--train-samples");
    const uint64_t seed = program.get<uint64_t>("--seed");
    const size_t loop = program.get<size_t>("--loop");
    const bool triangle = program.get<bool>("--triangle");
    const bool normalize = program.get<bool>("--normalize");
    const size_t warmups = program.get<size_t>("--warmup-loops");
    const auto nprobes = program.get<std::vector<size_t>>("--nprobes");
    if (k == 0 || block_size == 0 || loop == 0) {
        throw std::invalid_argument("k, block-size, and loop must be positive");
    }

    Index index;
    index.load_index(index_path);
    if (index.metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("The paper's PPD reproduction supports L2 only");
    }
    if (normalize) {
        // Cheap guard against accidentally attaching cosine queries to raw L2
        // base vectors. The source-bank identity is additionally bound by the driver.
        for (size_t lid = 0; lid < index.nlist; ++lid) {
            const IVF& list = index.lists[lid];
            if (!list.list_size) continue;
            const float* x = list.get_candidate_codes(0);
            double norm2 = 0;
            for (size_t axis = 0; axis < index.d; ++axis) norm2 += double(x[axis])*x[axis];
            if (std::abs(norm2-1.0) > 1e-4)
                throw std::invalid_argument("--normalize requires an already normalized IVF base index");
        }
    }
    PpdIndex ppd;
    const bool rebuild = program.get<bool>("--rebuild-sidecar");
    const bool no_save_sidecar = program.get<bool>("--no-save-sidecar");
    if (no_save_sidecar && program.get<bool>("--build-only")) {
        throw std::invalid_argument("--no-save-sidecar cannot be combined with --build-only");
    }
    if (!no_save_sidecar && !rebuild && std::filesystem::exists(sidecar_path)) {
        std::cout << "Loading PPD sidecar " << sidecar_path << std::endl;
        ppd.load(sidecar_path, index);
    } else {
        std::cout << std::format("Building PPD sidecar: samples={} d={} lists={}\n",
                                 train_samples, index.d, index.nlist);
        ppd.build(index, train_samples == 0 ? std::numeric_limits<size_t>::max() : train_samples, seed);
        if (no_save_sidecar) {
            std::cout << "Keeping PPD transformed database in memory; sidecar save disabled" << std::endl;
        } else {
            const std::string partial = sidecar_path + ".partial";
            ppd.save(partial);
            std::filesystem::rename(partial, sidecar_path);
            std::cout << "Saved PPD sidecar " << sidecar_path << std::endl;
        }
    }
    if (train_samples == 0) {
        size_t base_count = 0;
        for (size_t lid = 0; lid < index.nlist; ++lid) base_count += index.lists[lid].list_size;
        if (ppd.training_samples != base_count)
            throw std::invalid_argument("--train-samples 0 requires a full-training sidecar; rebuild the sampled sidecar");
    }
    if (program.get<bool>("--build-only")) return 0;

    auto [queries, available_nq, query_d] = loadXvecs(query_path);
    if (static_cast<size_t>(query_d) != index.d) {
        throw std::runtime_error("Query dimension does not match index");
    }
    size_t nq = program.get<size_t>("--nq");
    if (nq == 0) nq = available_nq;
    nq = std::min(nq, available_nq);
    if (nq == 0) throw std::invalid_argument("query file contains no queries");
    if (normalize) normalizeL2InPlace(queries.get(), nq, index.d);
    std::vector<idx_t> gt_ids;
    std::vector<float> gt_distances;
    ExternalGroundTruth external_gt;
    const bool external = gt_path.ends_with(".i32bin") || gt_path.ends_with(".ivecs");
    if (!gt_path.empty()) {
        if (external) {
            external_gt = load_external_groundtruth(gt_path, nq);
            if (k > external_gt.width) throw std::invalid_argument("k exceeds official groundtruth width");
        } else {
            gt_ids.resize(nq * k);
            gt_distances.resize(nq * k);
            loadResults(gt_path, gt_ids.data(), gt_distances.data(), nq, k);
        }
    }

    std::unique_ptr<CsvWriter> output;
    if (!csv_path.empty()) {
        output = std::make_unique<CsvWriter>(
            csv_path,
            std::vector<std::string>{
                "method", "nprobe", "block_size", "triangle", "training_samples",
                "query_time", "qps", "recall", "candidates", "triangle_pruned",
                "ppd_checked", "ppd_pruned", "full_distances", "dimensions",
                "equivalent_distance_computations", "ppd_prune_rate",
                "projection_us_per_query", "coarse_us_per_query",
                "verification_us_per_query", "overall_prune_rate", "n_query", "normalize",
                "measurement_loops", "warmup_loops", "benchmark_protocol", "benchmark_stats_enabled"},
            false, false);
    }
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    for (size_t nprobe : nprobes) {
        // Match query.cpp: exclude one warm-up pass when timing repeats.
        for (size_t warmup = 0; warmup < warmups; ++warmup) {
            searchPpd(index, ppd, nq, queries.get(), k, nprobe, block_size,
                      triangle, distances.data(), labels.data());
        }
        PpdSearchStats stats;
        double seconds = 0.0;
        double projection_seconds = 0.0;
        double coarse_seconds = 0.0;
        double verification_seconds = 0.0;
        for (size_t iteration = 0; iteration < loop; ++iteration) {
            Stopwatch search_watch;
            stats = searchPpd(index, ppd, nq, queries.get(), k, nprobe, block_size,
                              triangle, distances.data(), labels.data());
            seconds += search_watch.elapsedSeconds();
            projection_seconds += stats.projection_seconds;
            coarse_seconds += stats.coarse_seconds;
            verification_seconds += stats.verification_seconds;
        }
        seconds /= loop;
        projection_seconds /= loop;
        coarse_seconds /= loop;
        verification_seconds /= loop;
        const float recall = external ? calculate_id_recall(labels.data(), external_gt.ids.get(), nq, k, external_gt.width)
            : gt_ids.empty()
            ? std::numeric_limits<float>::quiet_NaN()
            : calculate_recall(labels.data(), distances.data(), gt_ids.data(),
                               gt_distances.data(), nq, k, MetricType::METRIC_L2);
        const double qps = nq / seconds;
        const double equivalent = static_cast<double>(stats.dimensions) / index.d;
        const double ppd_prune_rate = stats.ppd_checked == 0
            ? 0.0
            : static_cast<double>(stats.ppd_pruned) / stats.ppd_checked;
        const double projection_us = projection_seconds * 1e6 / nq;
        const double coarse_us = coarse_seconds * 1e6 / nq;
        const double verification_us = verification_seconds * 1e6 / nq;
        const double overall_prune = stats.candidates ?
            double(stats.triangle_pruned + stats.ppd_pruned)/stats.candidates : 0;
        std::cout << std::format(
            "PPD nprobe={} B={} triangle={} time={:.6f}s qps={:.3f} recall={:.6f} "
            "candidates={} tri_pruned={} ppd_pruned={} ppd_rate={:.4f} full={} dims={} "
            "equiv_dist={:.1f} projection={:.3f}us coarse={:.3f}us verify={:.3f}us\n",
            nprobe, block_size, triangle, seconds, qps, recall, stats.candidates,
            stats.triangle_pruned, stats.ppd_pruned, ppd_prune_rate,
            stats.full_distances, stats.dimensions, equivalent, projection_us,
            coarse_us, verification_us);
        if (output) {
            *output << "ppd" << nprobe << block_size << triangle << ppd.training_samples
                    << seconds << qps << recall << stats.candidates << stats.triangle_pruned
                    << stats.ppd_checked << stats.ppd_pruned << stats.full_distances
                    << stats.dimensions << equivalent << ppd_prune_rate << projection_us
                    << coarse_us << verification_us << overall_prune << nq << normalize
                    << loop << warmups << "search_only_v2"
#ifdef TRIBASE_ENABLE_STATS
                    << 1
#else
                    << 0
#endif
                    << std::endl;
        }
        const auto dump_path = program.get<std::string>("--dump-search-results");
        if (!dump_path.empty()) {
            prepareDirectory(dump_path);
            CsvWriter writer(dump_path, {"nprobe", "query_id", "neighbor", "id", "distance"},
                             nprobe != nprobes.front(), false);
            for (size_t qi = 0; qi < nq; ++qi)
                for (size_t ki = 0; ki < k; ++ki)
                    writer << nprobe << qi << ki << labels[qi*k+ki] << distances[qi*k+ki] << std::endl;
        }
    }
    return 0;
}
