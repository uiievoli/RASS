#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace tribase {

class Index;

// Reproduction of Liu et al. (Information Sciences 2018), progressive partial
// distance (PPD).  The sidecar deliberately stays outside the Tribase index
// format: it stores a global full-dimensional PCA transform and transformed
// candidates in the exact order of every IVF list.
class PpdIndex {
   public:
    size_t d = 0;
    size_t nlist = 0;
    size_t training_samples = 0;
    uint64_t seed = 0;
    std::vector<float> mean;
    // Row-major, descending principal components: basis[pc * d + axis].
    std::vector<float> basis;
    // Candidate-major transformed vectors, one array per IVF list.
    std::vector<std::vector<float>> transformed_lists;

    void build(const Index& index, size_t max_training_samples, uint64_t seed_value);
    void save(const std::string& path) const;
    void load(const std::string& path, const Index& index);
    void transform_query(const float* query, float* transformed) const;
};

struct PpdSearchStats {
    uint64_t candidates = 0;       // candidates before Triangle
    uint64_t triangle_pruned = 0;
    uint64_t ppd_checked = 0;      // candidates entering PPD
    uint64_t ppd_pruned = 0;       // early exits before dimension d
    uint64_t full_distances = 0;   // candidates reaching dimension d
    uint64_t dimensions = 0;       // total PCA dimensions accumulated
    double projection_seconds = 0;
    double coarse_seconds = 0;
    double verification_seconds = 0;
    double total_seconds = 0;
};

// L2-only IVF search. PPD is exact in real arithmetic and the accumulated
// d-dimensional value is reused as the final distance, as specified by the paper.
PpdSearchStats searchPpd(const Index& index,
                         const PpdIndex& ppd,
                         size_t query_count,
                         const float* queries,
                         size_t k,
                         size_t nprobe,
                         size_t block_size,
                         bool triangle,
                         float* distances,
                         idx_t* labels);

}  // namespace tribase
