#include "Microblock.h"
#include "MultiPivot.h"

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace {

std::vector<float> randomVectors(size_t count, size_t d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(count * d);
    for (float& v : values) v = dist(rng);
    return values;
}

}  // namespace

TEST(MicroblockTest, AabbBoundNeverExceedsPointPhiDistance) {
    using namespace tribase;
    constexpr size_t count = 128;
    constexpr size_t d = 16;
    constexpr size_t P = 8;
    auto points = randomVectors(count, d, 21);
    std::vector<float> centroid(d, 0.0f);
    PivotMetadata meta = selectPcaPivots(points.data(), count, d, P, centroid.data(), 0).metadata;
    computeCandidatePivotDistances(points.data(), count, d, meta);
    ASSERT_TRUE(meta.usable_signature(P));

    auto blocks = buildListMicroblocks(meta, d, P, /*bytes=*/2048, /*seed=*/7);
    ASSERT_FALSE(blocks.empty());
    for (const auto& block : blocks) {
        EXPECT_LE(block.members.size(), microblockVectorsPerBlock(d, 2048));
        EXPECT_EQ(block.lower.size(), P);
        EXPECT_EQ(block.upper.size(), P);
    }

    const float query[d] = {0.1f, -0.2f, 0.3f, 0.0f, -0.4f, 0.5f, -0.1f, 0.2f,
                            0.0f, 0.1f, -0.3f, 0.4f, -0.5f, 0.2f, 0.1f, -0.2f};
    std::array<double, MULTIPIVOT_MAX_PIVOTS> qdist{};
    std::array<float, MULTIPIVOT_MAX_PIVOTS> zq{};
    std::array<float, d> centered_query{};
    float srho = 0;
    ASSERT_TRUE(computePcaQuerySignature(query, d, meta, P, qdist.data(), zq.data(), &srho,
                                         centered_query.data()));
    std::array<float, MULTIPIVOT_MAX_PIVOTS> phi_q{};
    for (size_t t = 0; t + 1 < P; ++t) phi_q[t] = zq[t];
    phi_q[P - 1] = srho;

    for (const auto& block : blocks) {
        const float lb2 =
            aabbLowerBoundSquared(phi_q.data(), block.lower.data(), block.upper.data(), P);
        for (uint32_t member : block.members) {
            std::array<float, MULTIPIVOT_MAX_PIVOTS> phi_e{};
            ASSERT_TRUE(fillCandidatePhi(meta, member, P, phi_e.data()));
            float point2 = 0;
            for (size_t j = 0; j < P; ++j) {
                const float delta = phi_q[j] - phi_e[j];
                point2 += delta * delta;
            }
            EXPECT_LE(lb2, point2 + 1e-4f) << "member=" << member;
        }
    }
}

TEST(MicroblockTest, SiftLikeCapacity) {
    using namespace tribase;
    EXPECT_EQ(microblockVectorsPerBlock(128, 16384), 32u);
}
