#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

#include "SignaturePrune.h"
#include "common.h"

namespace tribase {

constexpr uint32_t MULTIPIVOT_FORMAT_VERSION = 11;
constexpr size_t MULTIPIVOT_MAX_PIVOTS = 512;

struct PivotMetadata {
    size_t requested_count = 0;
    size_t count = 0;
    size_t d = 0;
    std::vector<idx_t> source_ids;
    std::vector<float> codes;
    std::vector<double> pairwise_squared;
    // Candidate-major: candidate_pivot_squared[candidate * count + pivot].
    // Kept for max-mode pruning and signature construction.
    std::vector<double> candidate_pivot_squared;
    // Projection-only persisted center distance. Unlike candidate_pivot_squared,
    // this contains exactly one value per candidate.
    std::vector<float> candidate_centroid_squared;
    // Runtime-only residual at one configured prefix checkpoint. It is derived
    // from candidate_pivot_squared and signature_data after loading, so existing
    // indexes do not need to be rebuilt.
    std::vector<float> candidate_prefix_sqrt_rho;
    size_t candidate_prefix_length = 0;
    // Geometry for the configured P=count only.
    // (count - 1) x (count - 1), row-major.
    std::vector<double> gram_pinv;
    std::vector<double> gram_eigenvalues;
    size_t rank = 0;
    double condition = std::numeric_limits<double>::infinity();
    size_t fallback_count = 0;
    // True only for a global-projection list whose query geometry lives in
    // Index::global_pivots. It may retain candidate signatures without U.
    bool external_projection_geometry = false;

    // Dense (count-1)x(count-1) transform T with T^T T = G^+.
    // Used by generic affine-pivot methods only.
    std::vector<double> transform_T;

    // Fused PCA projection basis U = T * [u_1; ...; u_(P-1)], row-major
    // (count-1) x d. PCA-family query and candidate signatures use U directly,
    // so their runtime path has no separate (count-1)^2 transform.
    std::vector<float> U;

    // One SoA signature table for the configured pivot count P=count.
    // signature_data has length count * signature_candidate_count:
    //   value(dim, cand) = data[dim * candidate_count + cand]
    // dim in [0, count-2] is z; dim count-1 is the precomputed sqrt(rho).
    // When signature_precision != FLOAT32, float tables are cleared and compressed
    // tables below are used instead.
    std::vector<float> signature_data;
    std::vector<uint16_t> signature_data_f16;
    std::vector<int8_t> signature_data_i8;
    // Per dimension scale for int8 (decoded = stored * scale[dim]).
    std::vector<float> signature_i8_scales;
    SignaturePrecision signature_precision = SignaturePrecision::FLOAT32;
    size_t signature_candidate_count = 0;
    // Runtime-only accumulator used when an experiment advances through static
    // prefixes. It avoids rescanning coordinates [0,k) for every larger k.
    std::vector<double> candidate_prefix_residual_squared;

    void clear();
    bool usable_projection(size_t pivot_count) const;
    // Global geometry metadata may have T without per-candidate signature data.
    bool has_transform(size_t pivot_count) const;
    bool has_U(size_t pivot_count) const;
    bool usable_signature(size_t pivot_count) const;
    // Load one SoA component; optional max abs quantization error for conservative LB.
    float signatureValue(size_t active_count, size_t dim, size_t candidate_index,
                         float* max_abs_error = nullptr) const;
    void save(std::ostream& out) const;
    void load(std::istream& in, size_t expected_d, size_t expected_candidates);
};

struct PivotSelectionResult {
    PivotMetadata metadata;
    double elapsed_seconds = 0;
    // Descending covariance eigenvalues from PCA-family methods (empty otherwise).
    std::vector<double> pca_cov_eigenvalues;
    // IRLS L2,1 objective Σ α_i ||(I-UU^T)(x_i-c)|| per iteration (empty otherwise).
    std::vector<double> irls_objective_history;
    // Per accepted greedy direction: exact hard-pair prune gain G_t (build-time only).
    std::vector<double> greedy_gain_history;
};

bool validPivotCount(size_t count);
bool validMultipivotMethod(const std::string& method);
// PCA / weighted_pca / irls_pca / greedy_prune anchor on a fixed center.
bool isCentroidAnchoredMultipivotMethod(const std::string& method);

// Build-time-only params for greedy hard-pair direction selection (never updated at query).
struct GreedyPruneConfig {
    size_t pseudo_nq = 64;           // max pseudo-queries sampled per candidate pool (fallback)
    size_t negatives_per_q = 64;     // max negatives per training query
    size_t power_iters = 8;          // power-iteration steps for proxy eigvec
    size_t proxy_candidates = 4;     // #proxy directions evaluated with exact G_t
    size_t pair_direction_candidates = 4;  // extra normalized hard-pair r directions
    float boundary_alpha = 0.2f;     // prefer τ < ||q-x|| <= (1+α)τ
    double min_gain = 1.0;           // stop early if best G_t < min_gain (then pad)
    // Fraction of the full query set sampled upstream for real-query training (tag/logging).
    float real_query_ratio = 0.0f;
    // Per_list: assign each training query to this many nearest IVF lists (0 => nearest only).
    size_t real_lists_per_query = 8;
};

// If forced_v0 != nullptr, pivot 0 is that vector (e.g. IVF centroid) and the
// remaining pivots are chosen from candidates by affine FPS residual.
PivotSelectionResult selectAffineFps(const float* candidates,
                                     const idx_t* source_ids,
                                     size_t candidate_count,
                                     size_t d,
                                     size_t requested_count,
                                     uint64_t seed,
                                     const float* forced_v0 = nullptr,
                                     idx_t forced_v0_id = -1);

// PCA affine subspace: anchor μ = mean(candidates), or forced_mean when non-null
// (e.g. IVF list centroid for per_list). Directions = top (P-1) PCs of residuals.
PivotSelectionResult selectPcaPivots(const float* candidates,
                                     size_t candidate_count,
                                     size_t d,
                                     size_t requested_count,
                                     const float* forced_mean = nullptr,
                                     idx_t forced_mean_id = -1);

// Fixed-center weighted PCA: max_U Σ w_i ||U^T (x_i-c)||^2, U^T U = I.
// sample_weights == nullptr => uniform (same as selectPcaPivots with forced_mean).
// When forced_mean == nullptr, c defaults to the (weighted) data mean.
PivotSelectionResult selectWeightedPcaPivots(const float* candidates,
                                             size_t candidate_count,
                                             size_t d,
                                             size_t requested_count,
                                             const float* forced_mean = nullptr,
                                             idx_t forced_mean_id = -1,
                                             const float* sample_weights = nullptr);

// IRLS for min_U Σ α_i ||(I-UU^T)(x_i-c)|| via reweighted fixed-center PCA.
// Each iter: U <- weighted_pca(w); h_i <- residual; w_i <- α_i / max(h_i, ε).
PivotSelectionResult selectIrlsPcaPivots(const float* candidates,
                                         size_t candidate_count,
                                         size_t d,
                                         size_t requested_count,
                                         const float* forced_mean = nullptr,
                                         idx_t forced_mean_id = -1,
                                         size_t max_iter = 5,
                                         float residual_floor = 1e-6f,
                                         const float* alpha = nullptr);

// Build-time greedy directions. When train_queries!=nullptr && train_query_count>0,
// use those real queries (τ = 1-NN among candidates); else sample pseudo-queries from
// the pool. Query path never reselects pivots; result is centroid-anchored codes.
PivotSelectionResult selectGreedyPrunePivots(const float* candidates,
                                             size_t candidate_count,
                                             size_t d,
                                             size_t requested_count,
                                             const float* forced_mean = nullptr,
                                             idx_t forced_mean_id = -1,
                                             uint64_t seed = 0,
                                             const GreedyPruneConfig& config = {},
                                             const float* train_queries = nullptr,
                                             size_t train_query_count = 0);

void buildPivotGeometry(PivotMetadata& metadata);
void buildPcaProjectionU(PivotMetadata& metadata);

void computeCandidatePivotDistances(const float* candidates,
                                    size_t candidate_count,
                                    size_t d,
                                    PivotMetadata& metadata);

// Builds one Φ(e)=(z_e, sqrt(ρ_e)) float SoA table for metadata.count.
void computeCandidateSignatures(PivotMetadata& metadata,
                                size_t candidate_count);

// Compress float32 signature tables in-place to float16 / int8. No-op for float32.
void compressCandidateSignatures(PivotMetadata& metadata, SignaturePrecision precision);

// Prepare one SoA prefix residual cache. prefix_length=0 clears it.
void prepareCandidatePrefixResiduals(PivotMetadata& metadata, size_t prefix_length);

void computeQueryPivotDistances(const float* query,
                                size_t d,
                                const PivotMetadata& metadata,
                                size_t active_count,
                                double* squared_distances);

// Maps query-pivot distances into signature space. Also writes sqrt(ρ_q) as float.
bool computeQuerySignature(const PivotMetadata& metadata,
                           const double* query_pivot_squared,
                           size_t active_count,
                           float* z_query,
                           float* sqrt_rho_query);

// PCA-family path using the persisted fused projection basis U. Writes the
// float32 vector query-center into centered_query, computes its squared norm in
// the same pass, and then projects it with U. centered_query must contain d floats.
bool computePcaQuerySignature(const float* query,
                              size_t d,
                              const PivotMetadata& metadata,
                              size_t active_count,
                              double* query_pivot_squared,
                              float* z_query,
                              float* sqrt_rho_query,
                              float* centered_query);

double maxSinglePivotLowerBoundSquared(const double* query_pivot_squared,
                                       const double* candidate_pivot_squared,
                                       size_t count);

double affineProjectionLowerBoundSquared(const PivotMetadata& metadata,
                                         const double* query_pivot_squared,
                                         const double* candidate_pivot_squared,
                                         size_t active_count,
                                         bool* valid);

// Scalar bound (no early exit). Used by unit tests.
float signatureLowerBoundSquared(const float* z_query,
                                 float sqrt_rho_query,
                                 const PivotMetadata& metadata,
                                 size_t candidate_index,
                                 size_t active_count);

// Precision-aware path (float32/16/8) with conservative quantization slack.
bool signaturePrunes(const float* z_query,
                     float sqrt_rho_query,
                     const PivotMetadata& metadata,
                     size_t candidate_index,
                     size_t active_count,
                     float radius);

uint8_t signaturePruneMask8(const float* z_query,
                            float sqrt_rho_query,
                            const PivotMetadata& metadata,
                            size_t base_index,
                            size_t active_count,
                            float radius);

std::string joinPivotIds(const PivotMetadata& metadata, size_t active_count);
std::string joinPairwiseDistances(const PivotMetadata& metadata, size_t active_count);
std::string joinEigenvalues(const PivotMetadata& metadata, size_t active_count);

}  // namespace tribase
