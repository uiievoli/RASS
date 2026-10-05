#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>

#include "Index.h"

using namespace tribase;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <legacy_index> <output_v9_index> [scope=per_list] [P=16] [seed=6666]\n";
        return 1;
    }
    const std::string legacy_path = argv[1];
    const std::string output_path = argv[2];
    const std::string scope_name = argc > 3 ? argv[3] : "per_list";
    const size_t pivot_count = argc > 4 ? static_cast<size_t>(std::stoull(argv[4])) : 16;
    const uint64_t seed = argc > 5 ? std::stoull(argv[5]) : 6666ULL;

    MultiPivotScope scope = MultiPivotScope::PER_LIST;
    if (scope_name == "global") {
        scope = MultiPivotScope::GLOBAL;
    } else if (scope_name != "per_list") {
        std::cerr << "scope must be per_list or global\n";
        return 1;
    }

    Index index;
    std::cout << "Loading legacy index: " << legacy_path << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    index.load_legacy_index(legacy_path);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::format("Loaded d={} nlist={} added_opt={} sub_k={} in {:.2f}s\n",
                             index.d, index.nlist, static_cast<int>(index.added_opt_level),
                             index.sub_k, std::chrono::duration<double>(t1 - t0).count());

    // Preserve IVF list aux (triangle/subNN). For query --cache path matching when
    // --opt_levels OPT_NONE OPT_TRIANGLE (OR=1), stamp added_opt_level accordingly.
    index.added_opt_level = OptLevel::OPT_TRIANGLE;
    index.configure_multipivot(scope, "affine_fps", pivot_count, seed);
    std::cout << std::format("Building multipivot scope={} P={} seed={}\n",
                             multiPivotScopeName(scope), pivot_count, seed);
    auto t2 = std::chrono::steady_clock::now();
    index.rebuild_multipivot_metadata();
    auto t3 = std::chrono::steady_clock::now();
    std::cout << std::format(
        "Multipivot done selection={:.2f}s candidate_dist={:.2f}s total={:.2f}s comps={}\n",
        index.multipivot_selection_seconds, index.multipivot_candidate_distance_seconds,
        std::chrono::duration<double>(t3 - t2).count(),
        index.multipivot_build_distance_computations);

    index.save_index(output_path);
    std::cout << "Saved v6 index: " << output_path << std::endl;
    return 0;
}
