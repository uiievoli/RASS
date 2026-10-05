#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

namespace tribase {

#if defined(__GNUC__) || defined(__clang__)
#define TRIBASE_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define TRIBASE_ALWAYS_INLINE inline
#endif

// Float32 SoA lower-bound kernel. Keeping this definition visible to IVFScan avoids
// one function call per candidate batch in the search hot path.
TRIBASE_ALWAYS_INLINE bool signaturePrunes(const float* z_query,
                                            float sqrt_rho_query,
                                            const float* signature_soa,
                                            size_t candidate_index,
                                            size_t candidate_count,
                                            size_t active_count,
                                            float radius) {
    const size_t n = active_count - 1;
    const float gap =
        sqrt_rho_query - signature_soa[n * candidate_count + candidate_index];
    float score = gap * gap;
    if (score > radius) return true;
    for (size_t i = 0; i < n; ++i) {
        const float delta =
            z_query[i] - signature_soa[i * candidate_count + candidate_index];
        score += delta * delta;
        if (score > radius) return true;
    }
    return false;
}

// Evaluate eight candidates at once. Bit i is set when candidate base_index+i
// can be pruned. A radius captured at batch entry is conservative if the heap
// radius shrinks while accepted candidates are evaluated.
TRIBASE_ALWAYS_INLINE uint8_t signaturePruneMask8(const float* z_query,
                                                   float sqrt_rho_query,
                                                   const float* signature_soa,
                                                   size_t base_index,
                                                   size_t candidate_count,
                                                   size_t active_count,
                                                   float radius) {
#if defined(__AVX2__)
    const size_t n = active_count - 1;
    const __m256 radius_v = _mm256_set1_ps(radius);
    const __m256 query_rho_v = _mm256_set1_ps(sqrt_rho_query);
    const __m256 candidate_rho_v =
        _mm256_loadu_ps(signature_soa + n * candidate_count + base_index);
    const __m256 rho_gap = _mm256_sub_ps(query_rho_v, candidate_rho_v);
    __m256 score = _mm256_mul_ps(rho_gap, rho_gap);

    if (_mm256_movemask_ps(_mm256_cmp_ps(score, radius_v, _CMP_GT_OQ)) == 0xff) {
        return 0xff;
    }
    for (size_t i = 0; i < n; ++i) {
        const __m256 query_v = _mm256_set1_ps(z_query[i]);
        const __m256 candidate_v =
            _mm256_loadu_ps(signature_soa + i * candidate_count + base_index);
        const __m256 delta = _mm256_sub_ps(query_v, candidate_v);
        score = _mm256_fmadd_ps(delta, delta, score);
        if (_mm256_movemask_ps(_mm256_cmp_ps(score, radius_v, _CMP_GT_OQ)) == 0xff) {
            return 0xff;
        }
    }
    return static_cast<uint8_t>(
        _mm256_movemask_ps(_mm256_cmp_ps(score, radius_v, _CMP_GT_OQ)));
#else
    uint8_t mask = 0;
    for (unsigned lane = 0; lane < 8; ++lane) {
        if (signaturePrunes(z_query, sqrt_rho_query, signature_soa, base_index + lane,
                            candidate_count, active_count, radius)) {
            mask |= static_cast<uint8_t>(1u << lane);
        }
    }
    return mask;
#endif
}

// Fixed-block progressive kernel.  It evaluates one contiguous block of PCA
// coordinates, checks the residual lower bound, then continues with the next
// block only while at least one lane remains undecided.  The candidate center
// radii are stored once per candidate.
TRIBASE_ALWAYS_INLINE uint8_t blockSignaturePruneMask8(
    const float* z_query,
    float query_center_squared,
    const float* signature_soa,
    const float* candidate_center_squared,
    size_t base_index,
    size_t candidate_count,
    size_t active_count,
    size_t block_size,
    float radius,
    float precomputed_query_sqrt_rho = -1.0f) {
    if (active_count == 0) return 0;
    const size_t n = active_count - 1;
    if (n == 0) {
        const float hq = precomputed_query_sqrt_rho >= 0.0f
            ? precomputed_query_sqrt_rho
            : std::sqrt(std::max(0.0f, query_center_squared));
        uint8_t mask = 0;
        for (unsigned lane = 0; lane < 8; ++lane) {
            const float hx = std::sqrt(std::max(
                0.0f, candidate_center_squared[base_index + lane]));
            const float gap = hq - hx;
            if (gap * gap > radius) mask |= static_cast<uint8_t>(1u << lane);
        }
        return mask;
    }
    if (block_size == 0) block_size = n;
    // A list-level dynamic budget has one final prefix.  Its query residual is
    // independent of the candidate batch, so the caller computes it once.
    const bool use_precomputed_query_residual =
        precomputed_query_sqrt_rho >= 0.0f && block_size >= n;
#if defined(__AVX2__)
    const __m256 radius_v = _mm256_set1_ps(radius);
    __m256 score = _mm256_setzero_ps();
    __m256 hx2;
    {
        alignas(32) float initial[8];
        for (unsigned lane = 0; lane < 8; ++lane) {
            initial[lane] = std::max(
                0.0f, candidate_center_squared[base_index + lane]);
        }
        hx2 = _mm256_load_ps(initial);
    }
    float hq2 = use_precomputed_query_residual
        ? 0.0f : std::max(0.0f, query_center_squared);
    uint8_t pruned = 0;
    for (size_t begin = 0; begin < n; begin += block_size) {
        const size_t end = std::min(n, begin + block_size);
        for (size_t dim = begin; dim < end; ++dim) {
            const __m256 qv = _mm256_set1_ps(z_query[dim]);
            const __m256 xv = _mm256_loadu_ps(signature_soa + dim * candidate_count + base_index);
            const __m256 delta = _mm256_sub_ps(qv, xv);
            score = _mm256_fmadd_ps(delta, delta, score);
            hx2 = _mm256_sub_ps(hx2, _mm256_mul_ps(xv, xv));
            if (!use_precomputed_query_residual) {
                hq2 = std::max(0.0f, hq2 - z_query[dim] * z_query[dim]);
            }
        }
        const float hq_scalar = use_precomputed_query_residual
            ? precomputed_query_sqrt_rho
            : std::sqrt(std::max(0.0f, hq2));
        const __m256 hq = _mm256_set1_ps(hq_scalar);
        const __m256 hx = _mm256_sqrt_ps(_mm256_max_ps(hx2, _mm256_setzero_ps()));
        const __m256 gap = _mm256_sub_ps(hq, hx);
        const __m256 bound = _mm256_fmadd_ps(gap, gap, score);
        pruned = static_cast<uint8_t>(
            pruned | _mm256_movemask_ps(_mm256_cmp_ps(bound, radius_v, _CMP_GT_OQ)));
        if (pruned == 0xff) return pruned;
    }
    return pruned;
#else
    uint8_t mask = 0;
    for (unsigned lane = 0; lane < 8; ++lane) {
        const size_t candidate = base_index + lane;
        float score = 0.0f;
        float hq2 = use_precomputed_query_residual
            ? 0.0f : std::max(0.0f, query_center_squared);
        float hx2 = std::max(0.0f, candidate_center_squared[candidate]);
        for (size_t begin = 0; begin < n; begin += block_size) {
            const size_t end = std::min(n, begin + block_size);
            for (size_t dim = begin; dim < end; ++dim) {
                const float delta = z_query[dim] - signature_soa[dim * candidate_count + candidate];
                score += delta * delta;
                const float z = z_query[dim];
                const float x = signature_soa[dim * candidate_count + candidate];
                if (!use_precomputed_query_residual) {
                    hq2 = std::max(0.0f, hq2 - z * z);
                }
                hx2 = std::max(0.0f, hx2 - x * x);
            }
            const float hq = use_precomputed_query_residual
                ? precomputed_query_sqrt_rho : std::sqrt(hq2);
            const float gap = hq - std::sqrt(hx2);
            if (score + gap * gap > radius) {
                mask |= static_cast<uint8_t>(1u << lane);
                break;
            }
        }
    }
    return mask;
#endif
}

TRIBASE_ALWAYS_INLINE bool blockSignaturePrunes(
    const float* z_query,
    float query_center_squared,
    const float* signature_soa,
    const float* candidate_center_squared,
    size_t candidate_index,
    size_t candidate_count,
    size_t active_count,
    size_t block_size,
    float radius,
    float precomputed_query_sqrt_rho = -1.0f) {
    if (active_count == 0) return false;
    const size_t n = active_count - 1;
    if (n == 0) {
        const float hq = precomputed_query_sqrt_rho >= 0.0f
            ? precomputed_query_sqrt_rho
            : std::sqrt(std::max(0.0f, query_center_squared));
        const float gap = hq -
                          std::sqrt(std::max(
                              0.0f, candidate_center_squared[candidate_index]));
        return gap * gap > radius;
    }
    if (block_size == 0) block_size = n;
    const bool use_precomputed_query_residual =
        precomputed_query_sqrt_rho >= 0.0f && block_size >= n;
    float score = 0.0f;
    float hq2 = use_precomputed_query_residual
        ? 0.0f : std::max(0.0f, query_center_squared);
    float hx2 = std::max(0.0f, candidate_center_squared[candidate_index]);
    for (size_t begin = 0; begin < n; begin += block_size) {
        const size_t end = std::min(n, begin + block_size);
        for (size_t dim = begin; dim < end; ++dim) {
            const float q = z_query[dim];
            const float x = signature_soa[dim * candidate_count + candidate_index];
            const float delta = q - x;
            score += delta * delta;
            if (!use_precomputed_query_residual) {
                hq2 = std::max(0.0f, hq2 - q * q);
            }
            hx2 = std::max(0.0f, hx2 - x * x);
        }
        const float hq = use_precomputed_query_residual
            ? precomputed_query_sqrt_rho : std::sqrt(hq2);
        const float gap = hq - std::sqrt(hx2);
        if (score + gap * gap > radius) return true;
    }
    return false;
}

// Add one prefix checkpoint before the existing full-signature lower bound.
// After coordinates [0, k) have been accumulated, the unexplained norm is
// recovered from the saved distance to pivot 0:
//   rho_k(x) = ||x-c||^2 - sum_{i<k} z_i(x)^2.
// The prefix bound is sum_{i<k}(z_i(q)-z_i(e))^2 +
// (sqrt(rho_k(q))-sqrt(rho_k(e)))^2. If it does not prune, computation resumes
// with the precomputed full-P residual and the remaining projected coordinates.
TRIBASE_ALWAYS_INLINE bool stagedSignaturePrunes(const float* z_query,
                                                  float sqrt_rho_query,
                                                  float query_prefix_sqrt_rho,
                                                  const float* signature_soa,
                                                  const float* candidate_prefix_sqrt_rho,
                                                  size_t candidate_index,
                                                  size_t candidate_count,
                                                  size_t active_count,
                                                  size_t prefix_length,
                                                  float radius,
                                                  bool* prefix_pruned = nullptr) {
    if (prefix_pruned) *prefix_pruned = false;
    const size_t n = active_count - 1;
    const size_t prefix_count = std::min(prefix_length, n);
    float score = 0.0f;
    for (size_t i = 0; i < prefix_count; ++i) {
        const float q = z_query[i];
        const float e = signature_soa[i * candidate_count + candidate_index];
        const float delta = q - e;
        score += delta * delta;
    }
    const float prefix_gap =
        query_prefix_sqrt_rho - candidate_prefix_sqrt_rho[candidate_index];
    if (score + prefix_gap * prefix_gap > radius) {
        if (prefix_pruned) *prefix_pruned = true;
        return true;
    }
    // The checkpoint is already the final residual for this active prefix.
    // There is no later full-P stage to evaluate.
    if (prefix_count == n) return false;

    const float full_gap =
        sqrt_rho_query - signature_soa[n * candidate_count + candidate_index];
    score += full_gap * full_gap;
    if (score > radius) return true;
    for (size_t i = prefix_count; i < n; ++i) {
        const float delta =
            z_query[i] - signature_soa[i * candidate_count + candidate_index];
        score += delta * delta;
        if (score > radius) return true;
    }
    return false;
}

TRIBASE_ALWAYS_INLINE uint8_t stagedSignaturePruneMask8(
    const float* z_query,
    float sqrt_rho_query,
    float query_prefix_sqrt_rho,
    const float* signature_soa,
    const float* candidate_prefix_sqrt_rho,
    size_t base_index,
    size_t candidate_count,
    size_t active_count,
    size_t prefix_length,
    float radius,
    uint8_t* prefix_pruned_mask = nullptr) {
#if defined(__AVX2__)
    const size_t n = active_count - 1;
    const size_t prefix_count = std::min(prefix_length, n);
    const __m256 radius_v = _mm256_set1_ps(radius);
    __m256 score = _mm256_setzero_ps();
    for (size_t i = 0; i < prefix_count; ++i) {
        const float q = z_query[i];
        const __m256 query_v = _mm256_set1_ps(q);
        const __m256 candidate_v =
            _mm256_loadu_ps(signature_soa + i * candidate_count + base_index);
        const __m256 delta = _mm256_sub_ps(query_v, candidate_v);
        score = _mm256_fmadd_ps(delta, delta, score);
    }

    const __m256 prefix_gap = _mm256_sub_ps(
        _mm256_set1_ps(query_prefix_sqrt_rho),
        _mm256_loadu_ps(candidate_prefix_sqrt_rho + base_index));
    const __m256 prefix_bound = _mm256_fmadd_ps(prefix_gap, prefix_gap, score);
    uint8_t pruned = static_cast<uint8_t>(
        _mm256_movemask_ps(_mm256_cmp_ps(prefix_bound, radius_v, _CMP_GT_OQ)));
    if (prefix_pruned_mask) *prefix_pruned_mask = pruned;
    if (pruned == 0xff) return pruned;
    // The prefix bound is the complete bound when the checkpoint contains all
    // active coordinates. Avoid reading the Pmax residual row for a shorter
    // active prefix.
    if (prefix_count == n) return pruned;

    const __m256 query_rho_v = _mm256_set1_ps(sqrt_rho_query);
    const __m256 candidate_rho_v =
        _mm256_loadu_ps(signature_soa + n * candidate_count + base_index);
    const __m256 full_gap = _mm256_sub_ps(query_rho_v, candidate_rho_v);
    score = _mm256_fmadd_ps(full_gap, full_gap, score);
    pruned = static_cast<uint8_t>(
        pruned | _mm256_movemask_ps(_mm256_cmp_ps(score, radius_v, _CMP_GT_OQ)));
    if (pruned == 0xff) return pruned;

    for (size_t i = prefix_count; i < n; ++i) {
        const __m256 query_v = _mm256_set1_ps(z_query[i]);
        const __m256 candidate_v =
            _mm256_loadu_ps(signature_soa + i * candidate_count + base_index);
        const __m256 delta = _mm256_sub_ps(query_v, candidate_v);
        score = _mm256_fmadd_ps(delta, delta, score);
        pruned = static_cast<uint8_t>(
            pruned | _mm256_movemask_ps(_mm256_cmp_ps(score, radius_v, _CMP_GT_OQ)));
        if (pruned == 0xff) return pruned;
    }
    return pruned;
#else
    uint8_t mask = 0;
    uint8_t prefix_mask = 0;
    for (unsigned lane = 0; lane < 8; ++lane) {
        bool prefix_pruned = false;
        if (stagedSignaturePrunes(
                z_query, sqrt_rho_query, query_prefix_sqrt_rho, signature_soa,
                candidate_prefix_sqrt_rho, base_index + lane, candidate_count,
                active_count, prefix_length, radius, &prefix_pruned)) {
            mask |= static_cast<uint8_t>(1u << lane);
        }
        if (prefix_pruned) prefix_mask |= static_cast<uint8_t>(1u << lane);
    }
    if (prefix_pruned_mask) *prefix_pruned_mask = prefix_mask;
    return mask;
#endif
}

#undef TRIBASE_ALWAYS_INLINE

}  // namespace tribase
