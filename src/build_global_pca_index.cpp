#include "Index.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace tribase;

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "Usage: build_global_pca_index SOURCE_INDEX OUTPUT_INDEX PMAX");
        }
        const std::filesystem::path source(argv[1]);
        const std::filesystem::path output(argv[2]);
        const size_t pmax = std::stoull(argv[3]);
        if (!std::filesystem::is_regular_file(source)) {
            throw std::invalid_argument("source index does not exist");
        }
        if (std::filesystem::exists(output) ||
            std::filesystem::exists(output.string() + ".partial")) {
            throw std::invalid_argument("output or partial index already exists");
        }

        Index index;
        index.load_index(source.string());
        if (index.metric != MetricType::METRIC_L2 || pmax < 2 || pmax > index.d + 1 ||
            pmax > MULTIPIVOT_MAX_PIVOTS) {
            throw std::invalid_argument("Global PCA Pmax must be in [2,min(D+1,512)] for L2");
        }
        if (!(index.added_opt_level & OptLevel::OPT_TRIANGLE)) {
            throw std::invalid_argument("source index must contain Triangle metadata");
        }

        index.configure_multipivot(MultiPivotScope::GLOBAL, "pca", pmax, 0);
        index.upgrade_pruning(index.added_opt_level);
        if (index.global_pivots.count != pmax || !index.global_pivots.has_U(pmax)) {
            throw std::runtime_error("failed to construct the requested Global PCA bank");
        }
        const std::string partial = output.string() + ".partial";
        index.save_index(partial);
        std::error_code error;
        std::filesystem::rename(partial, output, error);
        if (error) {
            throw std::runtime_error("cannot publish rich index: " + error.message());
        }
        std::cout << "Built reusable Global PCA index: source=" << source
                  << " output=" << output << " Pmax=" << pmax
                  << " directions=" << pmax - 1 << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
