#include "Index.h"
#include "MultiPivot.h"
#include "SignaturePrune.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"

namespace {

double squaredL2(const float* a, const float* b, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        const double delta = static_cast<double>(a[i]) - b[i];
        sum += delta * delta;
    }
    return sum;
}

std::vector<float> randomVectors(size_t count, size_t d, uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    std::vector<float> values(count * d);
    for (float& value : values) value = distribution(generator);
    return values;
}

void expectSameResults(const std::vector<tribase::idx_t>& expected_ids,
                       const std::vector<float>& expected_distances,
                       const std::vector<tribase::idx_t>& actual_ids,
                       const std::vector<float>& actual_distances) {
    ASSERT_EQ(expected_ids.size(), actual_ids.size());
    ASSERT_EQ(expected_distances.size(), actual_distances.size());
    for (size_t i = 0; i < expected_ids.size(); ++i) {
        EXPECT_EQ(expected_ids[i], actual_ids[i]) << "result " << i;
        EXPECT_FLOAT_EQ(expected_distances[i], actual_distances[i]) << "result " << i;
    }
}

void runIndexEquivalence(tribase::MultiPivotScope scope) {
    using namespace tribase;
    constexpr size_t nb = 512;
    constexpr size_t nq = 32;
    constexpr size_t d = 12;
    constexpr size_t k = 10;
    constexpr size_t nlist = 16;

    std::vector<float> base = randomVectors(nb, d, 17);
    std::vector<float> queries = randomVectors(nq, d, 31);
    Index index(d, nlist, nlist, METRIC_L2, OPT_NONE, 0, 1, 1, false);
    index.configure_multipivot(scope, "pca", 4, 123);
    index.train(nb, base.data());
    index.add(nb, base.data());
    index.nprobe = nlist;

    std::vector<idx_t> baseline_ids(nq * k);
    std::vector<float> baseline_distances(nq * k);
    index.set_multipivot_search(MultiPivotMode::NONE, 0);
    index.search(nq, queries.data(), k, baseline_distances.data(), baseline_ids.data());

    for (MultiPivotMode mode : {MultiPivotMode::PROJECTION}) {
        std::vector<idx_t> ids(nq * k);
        std::vector<float> distances(nq * k);
        index.set_multipivot_search(mode, 4);
        Stats stats = index.search(nq, queries.data(), k, distances.data(), ids.data());
        expectSameResults(baseline_ids, baseline_distances, ids, distances);
#ifdef TRIBASE_ENABLE_STATS
        EXPECT_GT(stats.multipivot_checks, 0U);
        EXPECT_EQ(stats.multipivot_invalid, 0U);
#endif
    }

    index.set_projection_prefix_length(2);
    index.set_multipivot_search(MultiPivotMode::PROJECTION, 4);
    std::vector<idx_t> staged_ids(nq * k);
    std::vector<float> staged_distances(nq * k);
    index.search(nq, queries.data(), k, staged_distances.data(), staged_ids.data());
    expectSameResults(baseline_ids, baseline_distances, staged_ids, staged_distances);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        (scope == MultiPivotScope::GLOBAL
             ? "tribase_multipivot_global_test.index"
             : "tribase_multipivot_per_list_test.index");
    index.save_index(path.string());

    Index loaded;
    loaded.load_index(path.string());
    loaded.nprobe = nlist;
    loaded.set_projection_prefix_length(2);
    loaded.set_multipivot_search(MultiPivotMode::PROJECTION, 4);
    std::vector<idx_t> loaded_ids(nq * k);
    std::vector<float> loaded_distances(nq * k);
    loaded.search(nq, queries.data(), k, loaded_distances.data(), loaded_ids.data());
    expectSameResults(baseline_ids, baseline_distances, loaded_ids, loaded_distances);
    std::filesystem::remove(path);
}

}  // namespace

TEST(MultiPivotTest, PrecomputedQueryResidualMatchesBatchReconstruction) {
    using namespace tribase;
    constexpr size_t candidates = 8;
    constexpr size_t active_count = 7;
    constexpr size_t dimensions = active_count - 1;
    std::mt19937 generator(20260930);
    std::uniform_real_distribution<float> coordinate(-1.0f, 1.0f);
    std::uniform_real_distribution<float> residual(0.25f, 2.0f);
    std::uniform_real_distribution<float> threshold(0.0f, 20.0f);

    std::array<float, dimensions> query{};
    std::vector<float> signatures(dimensions * candidates);
    std::array<float, candidates> candidate_center_squared{};
    float query_center_squared = 0.0f;
    for (float& value : query) {
        value = coordinate(generator);
        query_center_squared += value * value;
    }
    const float query_residual = residual(generator);
    query_center_squared += query_residual * query_residual;
    for (size_t candidate = 0; candidate < candidates; ++candidate) {
        float squared = 0.0f;
        for (size_t dim = 0; dim < dimensions; ++dim) {
            const float value = coordinate(generator);
            signatures[dim * candidates + candidate] = value;
            squared += value * value;
        }
        const float tail = residual(generator);
        candidate_center_squared[candidate] = squared + tail * tail;
    }

    for (size_t trial = 0; trial < 128; ++trial) {
        const float radius = threshold(generator);
        const uint8_t reconstructed = blockSignaturePruneMask8(
            query.data(), query_center_squared, signatures.data(),
            candidate_center_squared.data(), 0, candidates, active_count,
            dimensions, radius);
        const uint8_t precomputed = blockSignaturePruneMask8(
            query.data(), query_center_squared, signatures.data(),
            candidate_center_squared.data(), 0, candidates, active_count,
            dimensions, radius, query_residual);
        EXPECT_EQ(reconstructed, precomputed);
        for (size_t candidate = 0; candidate < candidates; ++candidate) {
            EXPECT_EQ(
                blockSignaturePrunes(
                    query.data(), query_center_squared, signatures.data(),
                    candidate_center_squared.data(), candidate, candidates,
                    active_count, dimensions, radius),
                blockSignaturePrunes(
                    query.data(), query_center_squared, signatures.data(),
                    candidate_center_squared.data(), candidate, candidates,
                    active_count, dimensions, radius, query_residual));
        }
    }
}

TEST(MultiPivotTest, BoundsNeverExceedTrueL2) {
    using namespace tribase;
    constexpr size_t count = 128;
    constexpr size_t d = 10;
    std::vector<float> points = randomVectors(count, d, 7);
    std::vector<idx_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = static_cast<idx_t>(i);

    PivotMetadata metadata =
        selectAffineFps(points.data(), ids.data(), count, d, 8, 99).metadata;
    computeCandidatePivotDistances(points.data(), count, d, metadata);

    std::vector<float> queries = randomVectors(32, d, 11);
    for (size_t active_count : {8UL}) {
        std::vector<double> query_distances(active_count);
        for (size_t query = 0; query < 32; ++query) {
            computeQueryPivotDistances(queries.data() + query * d, d, metadata,
                                       active_count, query_distances.data());
            for (size_t candidate = 0; candidate < count; ++candidate) {
                const double* candidate_distances =
                    metadata.candidate_pivot_squared.data() + candidate * metadata.count;
                const double truth = squaredL2(
                    queries.data() + query * d, points.data() + candidate * d, d);
                const double max_bound = maxSinglePivotLowerBoundSquared(
                    query_distances.data(), candidate_distances, active_count);
                bool valid = false;
                const double projection_bound = affineProjectionLowerBoundSquared(
                    metadata, query_distances.data(), candidate_distances,
                    active_count, &valid);
                EXPECT_LE(max_bound, truth + 1e-8);
                ASSERT_TRUE(valid);
                EXPECT_LE(projection_bound, truth + 1e-8);
                if (active_count == 1) {
                    EXPECT_NEAR(projection_bound, max_bound, 1e-6);
                }
            }
        }
    }
}

TEST(MultiPivotTest, SignatureMatchesProjectionBound) {
    using namespace tribase;
    constexpr size_t count = 64;
    constexpr size_t d = 8;
    std::vector<float> points = randomVectors(count, d, 21);
    std::vector<idx_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = static_cast<idx_t>(i);
    PivotMetadata metadata =
        selectAffineFps(points.data(), ids.data(), count, d, 8, 3).metadata;
    computeCandidatePivotDistances(points.data(), count, d, metadata);
    ASSERT_TRUE(metadata.usable_signature(8));
    EXPECT_EQ(metadata.signature_data.size(), count * metadata.count);
    EXPECT_FALSE(metadata.usable_signature(4));

    std::vector<float> queries = randomVectors(16, d, 5);
    for (size_t active_count : {8UL}) {
        std::vector<double> query_distances(active_count);
        std::array<float, MULTIPIVOT_MAX_PIVOTS> z_query{};
        float sqrt_rho_query = 0;
        for (size_t query = 0; query < 16; ++query) {
            computeQueryPivotDistances(queries.data() + query * d, d, metadata,
                                       active_count, query_distances.data());
            ASSERT_TRUE(computeQuerySignature(metadata, query_distances.data(), active_count,
                                              z_query.data(), &sqrt_rho_query));
            for (size_t candidate = 0; candidate < count; ++candidate) {
                bool valid = false;
                const double slow = affineProjectionLowerBoundSquared(
                    metadata, query_distances.data(),
                    metadata.candidate_pivot_squared.data() + candidate * metadata.count,
                    active_count, &valid);
                ASSERT_TRUE(valid);
                const float fast = signatureLowerBoundSquared(
                    z_query.data(), sqrt_rho_query, metadata, candidate, active_count);
                EXPECT_NEAR(slow, static_cast<double>(fast), 1e-3)
                    << "P=" << active_count << " c=" << candidate;
                const double truth = squaredL2(queries.data() + query * d,
                                               points.data() + candidate * d, d);
                EXPECT_LE(fast, truth + 1e-3);
                // Early-exit prune must agree with comparing the scalar bound.
                const float radius = static_cast<float>(truth * 0.5);
                const bool prune = signaturePrunes(
                    z_query.data(), sqrt_rho_query,
                    metadata.signature_data.data(), candidate, count,
                    active_count, radius);
                EXPECT_EQ(prune, fast > radius);
            }
        }
    }
}

TEST(MultiPivotTest, ProjectionIncludesPerpendicularGap) {
    using namespace tribase;
    // Two pivots on the x-axis; query/candidate differ only in a perpendicular
    // coordinate so the old parallel-only bound is ~0 while the height gap is large.
    constexpr size_t d = 2;
    std::vector<float> pivots = {0.f, 0.f, 1.f, 0.f};
    std::vector<idx_t> ids = {0, 1};
    PivotMetadata metadata = selectAffineFps(pivots.data(), ids.data(), 2, d, 2, 0).metadata;
    ASSERT_EQ(metadata.count, 2U);

    const float query[2] = {0.5f, 4.0f};
    const float candidate[2] = {0.5f, 0.0f};
    std::vector<float> points = {candidate[0], candidate[1]};
    computeCandidatePivotDistances(points.data(), 1, d, metadata);
    std::vector<double> query_distances(2);
    computeQueryPivotDistances(query, d, metadata, 2, query_distances.data());

    bool valid = false;
    const double bound = affineProjectionLowerBoundSquared(
        metadata, query_distances.data(), metadata.candidate_pivot_squared.data(), 2, &valid);
    ASSERT_TRUE(valid);
    const double truth = squaredL2(query, candidate, d);
    EXPECT_LE(bound, truth + 1e-8);
    // Height gap alone is about 4^2 = 16; parallel component is near 0.
    EXPECT_GT(bound, 15.0);
}

TEST(MultiPivotTest, StagedPrefixResidualMatchesFullPcaDecision) {
    using namespace tribase;
    constexpr size_t count = 96;
    constexpr size_t d = 16;
    constexpr size_t p = 9;
    std::vector<float> points = randomVectors(count, d, 121);
    PivotMetadata metadata = selectPcaPivots(points.data(), count, d, p).metadata;
    computeCandidatePivotDistances(points.data(), count, d, metadata);
    ASSERT_TRUE(metadata.usable_signature(p));
    const std::array<size_t, 4> prefix_lengths = {1UL, 2UL, 4UL, 8UL};
    std::array<std::vector<float>, 4> prefix_residuals;
    for (size_t i = 0; i < prefix_lengths.size(); ++i) {
        prepareCandidatePrefixResiduals(metadata, prefix_lengths[i]);
        ASSERT_EQ(metadata.candidate_prefix_sqrt_rho.size(), count);
        prefix_residuals[i] = metadata.candidate_prefix_sqrt_rho;
    }

    std::vector<float> queries = randomVectors(12, d, 211);
    std::array<double, MULTIPIVOT_MAX_PIVOTS> query_distances{};
    std::array<float, MULTIPIVOT_MAX_PIVOTS> z_query{};
    std::array<float, d> centered{};
    for (size_t query = 0; query < 12; ++query) {
        float sqrt_rho = 0.0f;
        ASSERT_TRUE(computePcaQuerySignature(
            queries.data() + query * d, d, metadata, p, query_distances.data(),
            z_query.data(), &sqrt_rho, centered.data()));
        for (size_t candidate = 0; candidate < count; ++candidate) {
            const float truth = static_cast<float>(squaredL2(
                queries.data() + query * d, points.data() + candidate * d, d));
            for (float fraction : {0.25f, 0.5f, 0.9f, 1.1f}) {
                const float radius = truth * fraction;
                const bool full = signaturePrunes(
                    z_query.data(), sqrt_rho, metadata.signature_data.data(), candidate,
                    count, p, radius);
                for (size_t prefix_index = 0; prefix_index < prefix_lengths.size();
                     ++prefix_index) {
                    const size_t stride = prefix_lengths[prefix_index];
                    float query_prefix_remaining = static_cast<float>(query_distances[0]);
                    for (size_t dim = 0; dim < std::min(stride, p - 1); ++dim) {
                        query_prefix_remaining = std::max(
                            0.0f, query_prefix_remaining - z_query[dim] * z_query[dim]);
                    }
                    const bool staged = stagedSignaturePrunes(
                        z_query.data(), sqrt_rho, std::sqrt(query_prefix_remaining),
                        metadata.signature_data.data(),
                        prefix_residuals[prefix_index].data(), candidate, count, p,
                        stride, radius);
                    EXPECT_EQ(staged, full)
                        << "query=" << query << " candidate=" << candidate
                        << " stride=" << stride << " radius=" << radius;
                    if (staged) EXPECT_GT(truth, radius);
                }
            }
        }
    }
}

TEST(MultiPivotTest, DegenerateGeometryIsConservative) {
    using namespace tribase;
    constexpr size_t count = 16;
    constexpr size_t d = 4;
    std::vector<float> points(count * d, 0.0f);
    std::vector<idx_t> ids(count);
    for (size_t i = 0; i < count; ++i) {
        ids[i] = static_cast<idx_t>(i);
        points[i * d] = static_cast<float>(i % 4);
    }

    PivotMetadata metadata =
        selectAffineFps(points.data(), ids.data(), count, d, 4, 0).metadata;
    computeCandidatePivotDistances(points.data(), count, d, metadata);
    EXPECT_LT(metadata.rank, metadata.count - 1);

    const float query[d] = {1.5f, 2.0f, -1.0f, 0.5f};
    std::vector<double> query_distances(metadata.count);
    computeQueryPivotDistances(query, d, metadata, metadata.count,
                               query_distances.data());
    for (size_t candidate = 0; candidate < count; ++candidate) {
        bool valid = false;
        const double bound = affineProjectionLowerBoundSquared(
            metadata, query_distances.data(),
            metadata.candidate_pivot_squared.data() + candidate * metadata.count,
            metadata.count, &valid);
        ASSERT_TRUE(valid);
        EXPECT_LE(bound, squaredL2(query, points.data() + candidate * d, d) + 1e-8);
    }
}

TEST(MultiPivotTest, MetadataRoundTrip) {
    using namespace tribase;
    constexpr size_t count = 32;
    constexpr size_t d = 6;
    std::vector<float> points = randomVectors(count, d, 43);
    PivotMetadata original = selectPcaPivots(points.data(), count, d, 4).metadata;
    computeCandidatePivotDistances(points.data(), count, d, original);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    original.save(stream);
    stream.seekg(0);
    PivotMetadata loaded;
    loaded.load(stream, d, count);
    EXPECT_TRUE(loaded.source_ids.empty());
    ASSERT_EQ(loaded.codes.size(), d);
    EXPECT_TRUE(std::equal(loaded.codes.begin(), loaded.codes.end(), original.codes.begin()));
    EXPECT_TRUE(loaded.pairwise_squared.empty());
    EXPECT_TRUE(loaded.candidate_pivot_squared.empty());
    EXPECT_TRUE(loaded.gram_pinv.empty());
    EXPECT_TRUE(loaded.transform_T.empty());
    EXPECT_EQ(original.U, loaded.U);
    EXPECT_EQ(original.signature_data, loaded.signature_data);
    EXPECT_EQ(original.candidate_centroid_squared, loaded.candidate_centroid_squared);
    EXPECT_TRUE(loaded.has_U(2));
    EXPECT_TRUE(loaded.usable_signature(2));
    EXPECT_EQ(original.rank, loaded.rank);
    EXPECT_EQ(original.condition, loaded.condition);
    EXPECT_EQ(original.fallback_count, loaded.fallback_count);
}

TEST(MultiPivotTest, EmptyMetadataRoundTripForPopulatedList) {
    using namespace tribase;
    PivotMetadata original;
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    original.save(stream);
    stream.seekg(0);

    PivotMetadata loaded;
    EXPECT_NO_THROW(loaded.load(stream, 128, 37));
    EXPECT_EQ(loaded.count, 0U);
    EXPECT_EQ(loaded.signature_candidate_count, 0U);
    EXPECT_TRUE(loaded.signature_data.empty());
    EXPECT_TRUE(loaded.candidate_centroid_squared.empty());
}

TEST(MultiPivotTest, PcaForcedCentroidAndPrecomputedResidualTail) {
    using namespace tribase;
    constexpr size_t count = 80;
    constexpr size_t d = 8;
    std::vector<float> points = randomVectors(count, d, 19);
    std::vector<float> centroid(d, 0.0f);
    for (size_t j = 0; j < d; ++j) centroid[j] = static_cast<float>(j) * 0.1f;

    PivotMetadata metadata =
        selectPcaPivots(points.data(), count, d, 5, centroid.data(), 42).metadata;
    ASSERT_EQ(metadata.source_ids[0], 42);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(metadata.codes[j], centroid[j], 1e-6);
    }
    computeCandidatePivotDistances(points.data(), count, d, metadata);

    const float query[d] = {0.2f, -0.1f, 0.4f, 0.0f, -0.3f, 0.5f, -0.2f, 0.1f};
    std::array<double, MULTIPIVOT_MAX_PIVOTS> query_distances{};
    std::array<float, MULTIPIVOT_MAX_PIVOTS> z_query{};
    std::array<float, d> centered_query{};
    float sqrt_rho_query = 0;
    ASSERT_TRUE(computePcaQuerySignature(query, d, metadata, metadata.count,
                                         query_distances.data(), z_query.data(),
                                         &sqrt_rho_query, centered_query.data()));

    const size_t n = metadata.count - 1;
    for (size_t candidate = 0; candidate < count; ++candidate) {
        float z_norm2 = 0.0f;
        for (size_t row = 0; row < n; ++row) {
            const float z = metadata.signature_data[row * count + candidate];
            z_norm2 += z * z;
        }
        const double center_distance_squared =
            metadata.candidate_centroid_squared[candidate];
        const float expected_tail = static_cast<float>(
            std::sqrt(std::max(0.0, center_distance_squared - z_norm2)));
        EXPECT_NEAR(metadata.signature_data[n * count + candidate], expected_tail, 1e-6);

        const float bound = signatureLowerBoundSquared(
            z_query.data(), sqrt_rho_query, metadata, candidate, metadata.count);
        const double truth = squaredL2(query, points.data() + candidate * d, d);
        EXPECT_LE(bound, truth + 1e-3);

        const float radius = static_cast<float>(truth * 0.5);
        EXPECT_EQ(signaturePrunes(z_query.data(), sqrt_rho_query, metadata,
                                  candidate, metadata.count, radius),
                  bound > radius) << "candidate " << candidate;
    }

    // The general AVX2 mask must match scalar signaturePrunes lane-wise.
    {
        const float* soa = metadata.signature_data.data();
        const size_t nc = metadata.signature_candidate_count;
        ASSERT_GE(nc, 8u);
        for (float scale : {0.25f, 0.5f, 1.0f, 2.0f}) {
            for (size_t base = 0; base + 8 <= nc; base += 8) {
                const float radius =
                    static_cast<float>(squaredL2(query, points.data() + base * d, d) * scale);
                const uint8_t mask = signaturePruneMask8(z_query.data(), sqrt_rho_query, soa, base,
                                                        nc, metadata.count, radius);
                uint8_t expected = 0;
                for (int lane = 0; lane < 8; ++lane) {
                    if (signaturePrunes(z_query.data(), sqrt_rho_query, soa, base + lane, nc,
                                        metadata.count, radius)) {
                        expected |= static_cast<uint8_t>(1u << lane);
                    }
                }
                EXPECT_EQ(mask, expected) << "base=" << base << " scale=" << scale;
            }
        }
    }

}

TEST(MultiPivotTest, PcaAnchorIsMeanAndDirectionsUnit) {
    using namespace tribase;
    constexpr size_t count = 200;
    constexpr size_t d = 8;
    std::vector<float> points = randomVectors(count, d, 7);
    // Stretch first axis so PC1 is identifiable.
    for (size_t i = 0; i < count; ++i) points[i * d + 0] *= 8.0f;

    PivotMetadata metadata = selectPcaPivots(points.data(), count, d, 4).metadata;
    ASSERT_EQ(metadata.count, 4u);
    ASSERT_EQ(metadata.source_ids[0], -1);

    std::vector<double> mean(d, 0.0);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < d; ++j) mean[j] += points[i * d + j];
    }
    for (double& value : mean) value /= count;
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(metadata.codes[j], mean[j], 1e-4);
    }

    // Virtual pivots are μ and μ+u_i => ||v_i - μ|| = 1, directions orthonormal.
    for (size_t i = 1; i < metadata.count; ++i) {
        EXPECT_NEAR(std::sqrt(squaredL2(metadata.codes.data(),
                                        metadata.codes.data() + i * d, d)),
                    1.0, 1e-4);
        for (size_t j = i + 1; j < metadata.count; ++j) {
            double dot = 0;
            for (size_t k = 0; k < d; ++k) {
                const double ui = metadata.codes[i * d + k] - metadata.codes[k];
                const double uj = metadata.codes[j * d + k] - metadata.codes[k];
                dot += ui * uj;
            }
            EXPECT_NEAR(dot, 0.0, 1e-4) << "directions " << i << "," << j;
        }
    }

    // PC1 should align with axis 0.
    double abs_u0 = std::abs(metadata.codes[d + 0] - metadata.codes[0]);
    EXPECT_GT(abs_u0, 0.9);
}

TEST(MultiPivotTest, PcaProjectionBoundIsValid) {
    using namespace tribase;
    constexpr size_t count = 64;
    constexpr size_t d = 10;
    std::vector<float> points = randomVectors(count, d, 11);
    PivotMetadata metadata = selectPcaPivots(points.data(), count, d, 8).metadata;
    computeCandidatePivotDistances(points.data(), count, d, metadata);
    ASSERT_TRUE(metadata.usable_projection(metadata.count));

    const float query[d] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f, -0.9f, 1.0f};
    ASSERT_TRUE(metadata.candidate_pivot_squared.empty());
    std::array<double, MULTIPIVOT_MAX_PIVOTS> query_distances{};
    std::array<float, MULTIPIVOT_MAX_PIVOTS> query_signature{};
    std::array<float, d> centered_query{};
    float query_residual = 0.0f;
    ASSERT_TRUE(computePcaQuerySignature(
        query, d, metadata, metadata.count, query_distances.data(),
        query_signature.data(), &query_residual, centered_query.data()));
    for (size_t candidate = 0; candidate < count; ++candidate) {
        const double bound = signatureLowerBoundSquared(
            query_signature.data(), query_residual, metadata, candidate,
            metadata.count);
        EXPECT_LE(bound, squaredL2(query, points.data() + candidate * d, d) + 1e-5);
    }
}

TEST(MultiPivotTest, DirectPcaQuerySignatureMatchesGenericPath) {
    using namespace tribase;
    constexpr size_t count = 96;
    constexpr size_t d = 16;
    constexpr size_t P = 8;
    std::vector<float> points = randomVectors(count, d, 71);
    PivotMetadata metadata = selectPcaPivots(points.data(), count, d, P).metadata;
    computeCandidatePivotDistances(points.data(), count, d, metadata);
    ASSERT_EQ(metadata.U.size(), (P - 1) * d);
    ASSERT_TRUE(metadata.transform_T.empty());

    // Reconstruct the former generic T*b path as an independent reference.
    PivotMetadata generic_metadata = metadata;
    buildPivotGeometry(generic_metadata);
    ASSERT_TRUE(generic_metadata.U.empty());
    ASSERT_TRUE(generic_metadata.has_transform(P));
    computeCandidatePivotDistances(points.data(), count, d, generic_metadata);

    std::vector<float> queries = randomVectors(12, d, 73);
    std::array<float, d> centered_query{};
    for (size_t qi = 0; qi < 12; ++qi) {
        const float* query = queries.data() + qi * d;
        std::array<double, MULTIPIVOT_MAX_PIVOTS> generic_distances{};
        std::array<double, MULTIPIVOT_MAX_PIVOTS> direct_distances{};
        std::array<float, MULTIPIVOT_MAX_PIVOTS> generic_z{};
        std::array<float, MULTIPIVOT_MAX_PIVOTS> direct_z{};
        float generic_rho = 0;
        float direct_rho = 0;

        computeQueryPivotDistances(query, d, generic_metadata, P, generic_distances.data());
        ASSERT_TRUE(computeQuerySignature(generic_metadata, generic_distances.data(), P,
                                          generic_z.data(), &generic_rho));
        ASSERT_TRUE(computePcaQuerySignature(query, d, metadata, P,
                                             direct_distances.data(), direct_z.data(),
                                             &direct_rho, centered_query.data()));

        EXPECT_NEAR(direct_distances[0], generic_distances[0],
                    1e-5 * std::max(1.0, generic_distances[0]));
        EXPECT_NEAR(direct_rho, generic_rho, 1e-4);
        for (size_t axis = 0; axis < d; ++axis) {
            EXPECT_FLOAT_EQ(centered_query[axis], query[axis] - metadata.codes[axis]);
        }
        for (size_t i = 0; i + 1 < P; ++i) {
            EXPECT_NEAR(direct_z[i], generic_z[i], 1e-4) << "dimension " << i;
        }
        for (size_t candidate = 0; candidate < count; ++candidate) {
            const float direct_bound = signatureLowerBoundSquared(
                direct_z.data(), direct_rho, metadata, candidate, P);
            const float generic_bound = signatureLowerBoundSquared(
                generic_z.data(), generic_rho, generic_metadata, candidate, P);
            EXPECT_NEAR(direct_bound, generic_bound, 1e-3) << "candidate " << candidate;
            EXPECT_LE(direct_bound,
                      squaredL2(query, points.data() + candidate * d, d) + 1e-3);
        }
    }
}

TEST(MultiPivotTest, GlobalIndexMatchesBaselineAndRoundTrips) {
    runIndexEquivalence(tribase::MultiPivotScope::GLOBAL);
}

TEST(MultiPivotTest, PerListIndexMatchesBaselineAndRoundTrips) {
    runIndexEquivalence(tribase::MultiPivotScope::PER_LIST);
}

TEST(MultiPivotTest, PerListPcaPrecomputedResidualMatchesBaseline) {
    using namespace tribase;
    constexpr size_t nb = 512;
    constexpr size_t nq = 32;
    constexpr size_t d = 12;
    constexpr size_t k = 10;
    constexpr size_t nlist = 16;

    std::vector<float> base = randomVectors(nb, d, 41);
    std::vector<float> queries = randomVectors(nq, d, 43);
    // OPT_TRIANGLE uses its own centroid radii; MP stores precomputed residuals separately.
    Index index(d, nlist, nlist, METRIC_L2, OPT_TRIANGLE, 0, 1, 1, false);
    index.configure_multipivot(MultiPivotScope::PER_LIST, "pca", 6, 99);
    index.train(nb, base.data());
    index.add(nb, base.data());
    index.nprobe = nlist;

    for (size_t j = 0; j < d; ++j) {
        EXPECT_FLOAT_EQ(index.lists[0].pivots.codes[j], index.centroid_codes[j]);
    }

    std::vector<idx_t> baseline_ids(nq * k);
    std::vector<float> baseline_distances(nq * k);
    index.set_multipivot_search(MultiPivotMode::NONE, 0);
    index.search(nq, queries.data(), k, baseline_distances.data(), baseline_ids.data());

    std::vector<idx_t> ids(nq * k);
    std::vector<float> distances(nq * k);
    index.set_multipivot_search(MultiPivotMode::PROJECTION, 6);
    Stats stats = index.search(nq, queries.data(), k, distances.data(), ids.data());
    expectSameResults(baseline_ids, baseline_distances, ids, distances);
#ifdef TRIBASE_ENABLE_STATS
    EXPECT_GT(stats.multipivot_checks, 0U);
    EXPECT_EQ(stats.multipivot_invalid, 0U);
#endif
}

TEST(MultiPivotTest, WeightedPcaMatchesUniformPca) {
    using namespace tribase;
    constexpr size_t count = 64;
    constexpr size_t d = 10;
    constexpr size_t P = 5;
    std::vector<float> points = randomVectors(count, d, 77);
    std::vector<float> center(d, 0.25f);
    for (size_t j = 0; j < d; ++j) center[j] = static_cast<float>(j) * 0.05f;

    PivotMetadata pca = selectPcaPivots(points.data(), count, d, P, center.data(), 7).metadata;
    PivotMetadata weighted =
        selectWeightedPcaPivots(points.data(), count, d, P, center.data(), 7, nullptr).metadata;
    ASSERT_EQ(pca.codes.size(), weighted.codes.size());
    for (size_t i = 0; i < pca.codes.size(); ++i) {
        EXPECT_NEAR(pca.codes[i], weighted.codes[i], 1e-5) << "coord " << i;
    }
}

TEST(MultiPivotTest, IrlsPcaRunsAndKeepsFixedCenter) {
    using namespace tribase;
    constexpr size_t count = 64;
    constexpr size_t d = 10;
    constexpr size_t P = 5;
    std::vector<float> points = randomVectors(count, d, 91);
    std::vector<float> center(d, 0.0f);
    for (size_t j = 0; j < d; ++j) center[j] = static_cast<float>(j) * 0.07f;

    PivotSelectionResult irls =
        selectIrlsPcaPivots(points.data(), count, d, P, center.data(), 11, /*max_iter=*/5,
                            /*residual_floor=*/1e-6f, nullptr);
    ASSERT_EQ(irls.metadata.count, P);
    ASSERT_EQ(irls.metadata.source_ids[0], 11);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(irls.metadata.codes[j], center[j], 1e-6);
    }
    ASSERT_EQ(irls.irls_objective_history.size(), 5U);
    for (double obj : irls.irls_objective_history) {
        EXPECT_TRUE(std::isfinite(obj));
        EXPECT_GE(obj, 0.0);
    }
    // Objective is typically nonincreasing for IRLS L2,1 majorization (allow tiny noise).
    for (size_t t = 1; t < irls.irls_objective_history.size(); ++t) {
        EXPECT_LE(irls.irls_objective_history[t], irls.irls_objective_history[t - 1] + 1e-4);
    }
}

TEST(MultiPivotTest, GreedyPruneBuildTimeOnlyKeepsFixedCenter) {
    using namespace tribase;
    constexpr size_t count = 80;
    constexpr size_t d = 12;
    constexpr size_t P = 4;
    std::vector<float> points = randomVectors(count, d, 202);
    std::vector<float> center(d, 0.0f);
    for (size_t j = 0; j < d; ++j) center[j] = static_cast<float>(j) * 0.03f;

    GreedyPruneConfig cfg;
    cfg.pseudo_nq = 16;
    cfg.negatives_per_q = 16;
    cfg.power_iters = 6;
    cfg.proxy_candidates = 3;
    cfg.pair_direction_candidates = 2;
    cfg.boundary_alpha = 0.25f;
    cfg.min_gain = 1.0;

    PivotSelectionResult greedy =
        selectGreedyPrunePivots(points.data(), count, d, P, center.data(), 99, /*seed=*/7, cfg);
    ASSERT_EQ(greedy.metadata.count, P);
    ASSERT_EQ(greedy.metadata.source_ids[0], 99);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(greedy.metadata.codes[j], center[j], 1e-6);
    }
    // Directions must be length-1 offsets from the fixed center.
    for (size_t i = 1; i < P; ++i) {
        double norm2 = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const double delta =
                static_cast<double>(greedy.metadata.codes[i * d + j]) - center[j];
            norm2 += delta * delta;
        }
        EXPECT_NEAR(norm2, 1.0, 1e-4) << "direction " << i;
    }
    // Accepted gains recorded only at build time.
    for (double g : greedy.greedy_gain_history) {
        EXPECT_GE(g, cfg.min_gain - 1e-9);
    }
}

TEST(MultiPivotTest, GreedyPruneAcceptsRealTrainQueries) {
    using namespace tribase;
    constexpr size_t count = 64;
    constexpr size_t nq_train = 8;
    constexpr size_t d = 10;
    constexpr size_t P = 3;
    std::vector<float> points = randomVectors(count, d, 88);
    std::vector<float> queries = randomVectors(nq_train, d, 89);
    std::vector<float> center(d, 0.0f);

    GreedyPruneConfig cfg;
    cfg.pseudo_nq = 32;
    cfg.negatives_per_q = 16;
    cfg.power_iters = 4;
    cfg.proxy_candidates = 2;
    cfg.real_query_ratio = 0.05f;
    cfg.real_lists_per_query = 4;

    PivotSelectionResult real_q =
        selectGreedyPrunePivots(points.data(), count, d, P, center.data(), 1, /*seed=*/3, cfg,
                                queries.data(), nq_train);
    ASSERT_EQ(real_q.metadata.count, P);
    ASSERT_EQ(real_q.metadata.source_ids[0], 1);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(real_q.metadata.codes[j], center[j], 1e-6);
    }
}

TEST(MultiPivotTest, LbProbeWritesSummaryWithValidTightness) {
    using namespace tribase;
    constexpr size_t nb = 256;
    constexpr size_t nq = 16;
    constexpr size_t d = 8;
    constexpr size_t k = 1;
    constexpr size_t nlist = 8;
    constexpr size_t P = 4;

    std::vector<float> base = randomVectors(nb, d, 41);
    std::vector<float> queries = randomVectors(nq, d, 43);
    Index index(d, nlist, nlist, METRIC_L2, OPT_TRIANGLE, 0, 1, 1, false);
    index.configure_multipivot(MultiPivotScope::PER_LIST, "pca", P, 5);
    index.train(nb, base.data());
    index.add(nb, base.data());
    index.nprobe = nlist;
    index.set_multipivot_search(MultiPivotMode::PROJECTION, P);

    // Oracle tau^2 ≈ distance to exact 1-NN in the base (brute force).
    std::vector<float> tau2(nq, 0.0f);
    for (size_t qi = 0; qi < nq; ++qi) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < nb; ++i) {
            best = std::min(best, squaredL2(queries.data() + qi * d, base.data() + i * d, d));
        }
        tau2[qi] = static_cast<float>(best);
    }

    const std::filesystem::path raw =
        std::filesystem::temp_directory_path() / "tribase_lb_probe_test.csv";
    Index::LbProbeConfig cfg;
    cfg.max_probes = nlist;
    cfg.max_candidates_per_list = 32;
    cfg.max_raw_rows = 5000;
    cfg.triangle_survivors_only = true;
    cfg.boundary_alpha = 0.2f;
    index.probe_lower_bounds(nq, queries.data(), tau2.data(), raw.string(), cfg);

    const std::filesystem::path summary = raw.string() + ".summary.csv";
    ASSERT_TRUE(std::filesystem::exists(raw));
    ASSERT_TRUE(std::filesystem::exists(summary));

    std::ifstream in(summary);
    ASSERT_TRUE(in);
    std::string header;
    ASSERT_TRUE(std::getline(in, header));
    EXPECT_NE(header.find("mean_rho"), std::string::npos);
    size_t data_rows = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ++data_rows;
        // mean_rho is column index 7 in the summary schema.
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cols;
        while (std::getline(ss, cell, ',')) cols.push_back(cell);
        ASSERT_GE(cols.size(), 9u);
        const double mean_rho = std::stod(cols[7]);
        EXPECT_GE(mean_rho, 0.0);
        EXPECT_LE(mean_rho, 1.0 + 1e-3);
    }
    EXPECT_EQ(data_rows, 1U);

    std::error_code ec;
    std::filesystem::remove(raw, ec);
    std::filesystem::remove(summary, ec);
}
