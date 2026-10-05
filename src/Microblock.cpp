#include "Microblock.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace tribase {
namespace {

template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

void writeFloatVector(std::ostream& out, const std::vector<float>& values) {
    const uint64_t n = values.size();
    writeScalar<uint64_t>(out, n);
    if (n) out.write(reinterpret_cast<const char*>(values.data()), n * sizeof(float));
}

void readFloatVector(std::istream& in, std::vector<float>& values) {
    const uint64_t n = readScalar<uint64_t>(in);
    values.resize(static_cast<size_t>(n));
    if (n) in.read(reinterpret_cast<char*>(values.data()), n * sizeof(float));
}

void writeU32Vector(std::ostream& out, const std::vector<uint32_t>& values) {
    const uint64_t n = values.size();
    writeScalar<uint64_t>(out, n);
    if (n) out.write(reinterpret_cast<const char*>(values.data()), n * sizeof(uint32_t));
}

void readU32Vector(std::istream& in, std::vector<uint32_t>& values) {
    const uint64_t n = readScalar<uint64_t>(in);
    values.resize(static_cast<size_t>(n));
    if (n) in.read(reinterpret_cast<char*>(values.data()), n * sizeof(uint32_t));
}

void accumulateAabb(MicroblockMeta& block, const float* phi, size_t dim) {
    if (block.lower.empty()) {
        block.lower.assign(phi, phi + dim);
        block.upper.assign(phi, phi + dim);
        return;
    }
    for (size_t j = 0; j < dim; ++j) {
        block.lower[j] = std::min(block.lower[j], phi[j]);
        block.upper[j] = std::max(block.upper[j], phi[j]);
    }
}

MicroblockMeta makeBlockFromMembers(const std::vector<float>& phis,
                                    size_t dim,
                                    size_t d,
                                    const std::vector<uint32_t>& members) {
    MicroblockMeta block;
    block.members = members;
    block.vector_count = static_cast<uint32_t>(members.size());
    block.payload_length = static_cast<uint32_t>(members.size() * d * sizeof(float));
    for (uint32_t idx : members) {
        accumulateAabb(block, phis.data() + static_cast<size_t>(idx) * dim, dim);
    }
    return block;
}

}  // namespace

float distToInterval(float x, float lo, float hi) {
    if (x < lo) return lo - x;
    if (x > hi) return x - hi;
    return 0.0f;
}

float aabbLowerBoundSquared(const float* q,
                            const float* lower,
                            const float* upper,
                            size_t dim) {
    float sum = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
        const float delta = distToInterval(q[j], lower[j], upper[j]);
        sum += delta * delta;
    }
    return sum;
}

bool fillCandidatePhi(const PivotMetadata& metadata,
                      size_t candidate_index,
                      size_t P,
                      float* phi_out) {
    if (P == 0 || !metadata.usable_signature(P) || candidate_index >= metadata.signature_candidate_count) {
        return false;
    }
    if (P == 1) {
        phi_out[0] = metadata.signatureValue(P, 0, candidate_index);
        return true;
    }
    const size_t n = P - 1;
    for (size_t i = 0; i < n; ++i) {
        const float z = metadata.signatureValue(P, i, candidate_index);
        phi_out[i] = z;
    }
    phi_out[n] = metadata.signatureValue(P, n, candidate_index);
    return true;
}

std::vector<MicroblockMeta> buildListMicroblocks(const PivotMetadata& metadata,
                                                 size_t d,
                                                 size_t built_P,
                                                 size_t bytes,
                                                 uint64_t seed,
                                                 int kmeans_iters) {
    std::vector<MicroblockMeta> blocks;
    if (built_P == 0 || !metadata.usable_signature(built_P)) return blocks;
    const size_t n = metadata.signature_candidate_count;
    if (n == 0) return blocks;

    const size_t dim = built_P;
    const size_t capacity = microblockVectorsPerBlock(d, bytes);
    std::vector<float> phis(n * dim);
    for (size_t i = 0; i < n; ++i) {
        if (!fillCandidatePhi(metadata, i, built_P, phis.data() + i * dim)) {
            throw std::runtime_error("Failed to materialize candidate signature for microblocks");
        }
    }

    const size_t k = std::max<size_t>(1, (n + capacity - 1) / capacity);
    std::vector<uint32_t> assign(n, 0);
    std::vector<float> centers(k * dim, 0.0f);
    std::mt19937 rng(static_cast<uint32_t>(seed ^ (n * 1315423911u) ^ (k * 2654435761u)));

    // Init centers: spread unique samples.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    for (size_t i = 0; i < n; ++i) {
        std::uniform_int_distribution<size_t> dist(i, n - 1);
        std::swap(order[i], order[dist(rng)]);
    }
    for (size_t c = 0; c < k; ++c) {
        const size_t src = order[c % n];
        std::copy_n(phis.data() + src * dim, dim, centers.data() + c * dim);
    }

    std::vector<size_t> counts(k, 0);
    for (int iter = 0; iter < kmeans_iters; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            const float* x = phis.data() + i * dim;
            float best = std::numeric_limits<float>::infinity();
            uint32_t best_c = 0;
            for (size_t c = 0; c < k; ++c) {
                const float* mu = centers.data() + c * dim;
                float dist2 = 0.0f;
                for (size_t j = 0; j < dim; ++j) {
                    const float delta = x[j] - mu[j];
                    dist2 += delta * delta;
                }
                if (dist2 < best) {
                    best = dist2;
                    best_c = static_cast<uint32_t>(c);
                }
            }
            assign[i] = best_c;
        }
        std::fill(centers.begin(), centers.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            const size_t c = assign[i];
            float* mu = centers.data() + c * dim;
            const float* x = phis.data() + i * dim;
            for (size_t j = 0; j < dim; ++j) mu[j] += x[j];
            counts[c]++;
        }
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                std::uniform_int_distribution<size_t> dist(0, n - 1);
                const size_t src = dist(rng);
                std::copy_n(phis.data() + src * dim, dim, centers.data() + c * dim);
                continue;
            }
            const float inv = 1.0f / static_cast<float>(counts[c]);
            float* mu = centers.data() + c * dim;
            for (size_t j = 0; j < dim; ++j) mu[j] *= inv;
        }
    }

    std::vector<std::vector<uint32_t>> groups(k);
    for (size_t i = 0; i < n; ++i) groups[assign[i]].push_back(static_cast<uint32_t>(i));

    // Sort each group by first Φ dim for stable packing, then split to capacity.
    for (auto& group : groups) {
        if (group.empty()) continue;
        std::sort(group.begin(), group.end(), [&](uint32_t a, uint32_t b) {
            return phis[static_cast<size_t>(a) * dim] < phis[static_cast<size_t>(b) * dim];
        });
        for (size_t offset = 0; offset < group.size(); offset += capacity) {
            const size_t take = std::min(capacity, group.size() - offset);
            std::vector<uint32_t> members(group.begin() + static_cast<std::ptrdiff_t>(offset),
                                          group.begin() + static_cast<std::ptrdiff_t>(offset + take));
            blocks.push_back(makeBlockFromMembers(phis, dim, d, members));
        }
    }
    return blocks;
}

void saveMicroblocks(std::ostream& out, const std::vector<MicroblockMeta>& blocks) {
    writeScalar<uint32_t>(out, MICROBLOCK_BLOCK_MAGIC);
    writeScalar<uint64_t>(out, blocks.size());
    for (const auto& block : blocks) {
        writeFloatVector(out, block.lower);
        writeFloatVector(out, block.upper);
        writeU32Vector(out, block.members);
        writeScalar<uint32_t>(out, block.vector_count);
        writeScalar<uint32_t>(out, block.payload_length);
    }
}

bool loadMicroblocks(std::istream& in, std::vector<MicroblockMeta>& blocks) {
    blocks.clear();
    if (!in || in.peek() == EOF) return false;
    const auto pos = in.tellg();
    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!in || magic != MICROBLOCK_BLOCK_MAGIC) {
        in.clear();
        in.seekg(pos);
        return false;
    }
    const uint64_t count = readScalar<uint64_t>(in);
    blocks.resize(static_cast<size_t>(count));
    for (auto& block : blocks) {
        readFloatVector(in, block.lower);
        readFloatVector(in, block.upper);
        readU32Vector(in, block.members);
        block.vector_count = readScalar<uint32_t>(in);
        block.payload_length = readScalar<uint32_t>(in);
        if (block.lower.size() != block.upper.size() ||
            block.vector_count != block.members.size()) {
            throw std::runtime_error("Corrupt microblock metadata");
        }
    }
    return true;
}

}  // namespace tribase
