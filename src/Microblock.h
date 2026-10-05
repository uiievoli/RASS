#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

#include "MultiPivot.h"

namespace tribase {

constexpr size_t MICROBLOCK_DEFAULT_BYTES = 16384;
constexpr uint32_t MICROBLOCK_BLOCK_MAGIC = 0x4d424c4bu;  // 'MBLK'

inline size_t microblockVectorsPerBlock(size_t d, size_t bytes = MICROBLOCK_DEFAULT_BYTES) {
    if (d == 0) return 1;
    const size_t per = bytes / (d * sizeof(float));
    return per == 0 ? 1 : per;
}

struct MicroblockMeta {
    // AABB in comparable Φ space for built_P: (z_0..z_{P-2}, sqrt(ρ)).
    std::vector<float> lower;
    std::vector<float> upper;
    std::vector<uint32_t> members;  // indices into the IVF list
    uint32_t vector_count = 0;
    uint32_t payload_length = 0;  // vector_count * d * sizeof(float)
};

float distToInterval(float x, float lo, float hi);

// Sum of squared distances from q[0..dim) to the axis-aligned box.
float aabbLowerBoundSquared(const float* q,
                            const float* lower,
                            const float* upper,
                            size_t dim);

// Write Φ(e) for candidate into phi_out[0..P): z then sqrt(ρ) (reconstruct if needed).
bool fillCandidatePhi(const PivotMetadata& metadata,
                      size_t candidate_index,
                      size_t P,
                      float* phi_out);

// Build list microblocks by k-means in Φ_{built_P} space, then split clusters to
// at most vectors_per_block members (16KB-oriented).
std::vector<MicroblockMeta> buildListMicroblocks(const PivotMetadata& metadata,
                                                 size_t d,
                                                 size_t built_P,
                                                 size_t bytes = MICROBLOCK_DEFAULT_BYTES,
                                                 uint64_t seed = 6666,
                                                 int kmeans_iters = 15);

void saveMicroblocks(std::ostream& out, const std::vector<MicroblockMeta>& blocks);
// Returns false if no microblock trailer (legacy IVF); leaves blocks empty.
bool loadMicroblocks(std::istream& in, std::vector<MicroblockMeta>& blocks);

}  // namespace tribase
