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
    const auto nprobes = program.get<std::vector<size_t>>("--nprobes");
    if (k == 0 || block_size == 0 || loop == 0) {
        throw std::invalid_argument("k, block-size, and loop must be positive");
    }

    Index index;
    index.load_index(index_path);
    if (index.metric != MetricType::METRIC_L2) {
        throw std::invalid_argument("The paper's PPD reproduction supports L2 only");
    }
    PpdIndex ppd;
    const bool rebuild = program.get<bool>("--rebuild-sidecar");
    if (!rebuild && std::filesystem::exists(sidecar_path)) {
        std::cout << "Loading PPD sidecar " << sidecar_path << std::endl;
        ppd.load(sidecar_path, index);
    } else {
        std::cout << std::format("Building PPD sidecar: samples={} d={} lists={}\n",
                                 train_samples, index.d, index.nlist);
        ppd.build(index, train_samples, seed);
        const std::string partial = sidecar_path + ".partial";
        ppd.save(partial);
        std::filesystem::rename(partial, sidecar_path);
        std::cout << "Saved PPD sidecar " << sidecar_path << std::endl;
    }

    auto [queries, available_nq, query_d] = loadXvecs(query_path);
    if (static_cast<size_t>(query_d) != index.d) {
        throw std::runtime_error("Query dimension does not match index");
    }
    size_t nq = program.get<size_t>("--nq");
    if (nq == 0) nq = available_nq;
    nq = std::min(nq, available_nq);
    std::vector<idx_t> gt_ids;
    std::vector<float> gt_distances;
    if (!gt_path.empty()) {
        gt_ids.resize(nq * k);
        gt_distances.resize(nq * k);
        loadResults(gt_path, gt_ids.data(), gt_distances.data(), nq, k);
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
                "verification_us_per_query"},
            false, false);
    }
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    for (size_t nprobe : nprobes) {
        // Match query.cpp: exclude one warm-up pass when timing repeats.
        if (loop > 1) {
            searchPpd(index, ppd, nq, queries.get(), k, nprobe, block_size,
                      triangle, distances.data(), labels.data());
        }
        PpdSearchStats stats;
        double seconds = 0.0;
        double projection_seconds = 0.0;
        double coarse_seconds = 0.0;
        double verification_seconds = 0.0;
        for (size_t iteration = 0; iteration < loop; ++iteration) {
            stats = searchPpd(index, ppd, nq, queries.get(), k, nprobe, block_size,
                              triangle, distances.data(), labels.data());
            seconds += stats.total_seconds;
            projection_seconds += stats.projection_seconds;
            coarse_seconds += stats.coarse_seconds;
            verification_seconds += stats.verification_seconds;
        }
        seconds /= loop;
        projection_seconds /= loop;
        coarse_seconds /= loop;
        verification_seconds /= loop;
        const float recall = gt_ids.empty()
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
                    << coarse_us << verification_us << std::endl;
        }
    }
    return 0;
}
