#include "MultiPivot.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>

#include "utils.h"

namespace tribase {
namespace {

constexpr uint32_t PIVOT_BLOCK_MAGIC = 0x50565431;  // PVT1
constexpr uint64_t MAX_SERIALIZED_ELEMENTS = (1ULL << 34);

template <typename T>
void writeScalar(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("Failed to write pivot metadata");
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("Truncated pivot metadata");
    return value;
}

template <typename T>
void writeVector(std::ostream& out, const std::vector<T>& values) {
    writeScalar<uint64_t>(out, values.size());
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
        if (!out) throw std::runtime_error("Failed to write pivot metadata vector");
    }
}

template <typename T>
void readVector(std::istream& in, std::vector<T>& values, uint64_t expected = std::numeric_limits<uint64_t>::max()) {
    const uint64_t size = readScalar<uint64_t>(in);
    if (size > MAX_SERIALIZED_ELEMENTS || (expected != std::numeric_limits<uint64_t>::max() && size != expected)) {
        throw std::runtime_error("Invalid pivot metadata vector size");
    }
    values.resize(size);
    if (size != 0) {
        in.read(reinterpret_cast<char*>(values.data()), size * sizeof(T));
        if (!in) throw std::runtime_error("Truncated pivot metadata vector");
    }
}

double squaredL2(const float* a, const float* b, size_t d) {
    double result = 0;
#pragma omp simd reduction(+ : result)
    for (size_t j = 0; j < d; ++j) {
        const double delta = static_cast<double>(a[j]) - static_cast<double>(b[j]);
        result += delta * delta;
    }
    return result;
}

struct EigenResult {
    std::vector<double> values;
    std::vector<double> vectors;
};

EigenResult symmetricJacobi(std::vector<double> matrix, size_t n) {
    EigenResult result;
    result.values.resize(n);
    result.vectors.assign(n * n, 0);
    for (size_t i = 0; i < n; ++i) result.vectors[i * n + i] = 1;
    if (n == 0) return result;

    const size_t max_iterations = std::max<size_t>(64, 64 * n * n);
    for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
        size_t p = 0, q = 0;
        double largest = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double value = std::abs(matrix[i * n + j]);
                if (value > largest) {
                    largest = value;
                    p = i;
                    q = j;
                }
            }
        }
        double scale = 0;
        for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::abs(matrix[i * n + i]));
        if (largest <= std::max(1.0, scale) * 1e-14) break;

        const double app = matrix[p * n + p];
        const double aqq = matrix[q * n + q];
        const double apq = matrix[p * n + q];
        const double angle = 0.5 * std::atan2(2 * apq, aqq - app);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        for (size_t k = 0; k < n; ++k) {
            const double mkp = matrix[k * n + p];
            const double mkq = matrix[k * n + q];
            matrix[k * n + p] = c * mkp - s * mkq;
            matrix[k * n + q] = s * mkp + c * mkq;
        }
        for (size_t k = 0; k < n; ++k) {
            const double mpk = matrix[p * n + k];
            const double mqk = matrix[q * n + k];
            matrix[p * n + k] = c * mpk - s * mqk;
            matrix[q * n + k] = s * mpk + c * mqk;
        }
        matrix[p * n + q] = matrix[q * n + p] = 0;
        for (size_t k = 0; k < n; ++k) {
            const double vkp = result.vectors[k * n + p];
            const double vkq = result.vectors[k * n + q];
            result.vectors[k * n + p] = c * vkp - s * vkq;
            result.vectors[k * n + q] = s * vkp + c * vkq;
        }
    }
    for (size_t i = 0; i < n; ++i) result.values[i] = matrix[i * n + i];

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return result.values[a] > result.values[b];
    });
    EigenResult sorted;
    sorted.values.resize(n);
    sorted.vectors.resize(n * n);
    for (size_t col = 0; col < n; ++col) {
        sorted.values[col] = result.values[order[col]];
        for (size_t row = 0; row < n; ++row) {
            sorted.vectors[row * n + col] = result.vectors[row * n + order[col]];
        }
    }
    return sorted;
}

std::vector<double> gramForPrefix(const PivotMetadata& metadata, size_t p) {
    const size_t n = p - 1;
    std::vector<double> gram(n * n);
    for (size_t i = 0; i < n; ++i) {
        const double vi2 = metadata.pairwise_squared[i + 1];
        for (size_t j = 0; j < n; ++j) {
            const double vj2 = metadata.pairwise_squared[j + 1];
            const double dij2 = metadata.pairwise_squared[(i + 1) * metadata.count + j + 1];
            gram[i * n + j] = 0.5 * (vi2 + vj2 - dij2);
        }
    }
    return gram;
}

// In-place Cholesky G := L (lower) with G = L L^T. Returns false if not SPD.
bool choleskyLower(std::vector<double>& G, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = G[i * n + j];
            for (size_t k = 0; k < j; ++k) sum -= G[i * n + k] * G[j * n + k];
            if (i == j) {
                if (sum <= 1e-14) return false;
                G[i * n + i] = std::sqrt(sum);
            } else {
                G[i * n + j] = sum / G[j * n + j];
            }
        }
        for (size_t j = i + 1; j < n; ++j) G[i * n + j] = 0;
    }
    return true;
}

// Invert lower-triangular L into dense T = L^{-1}.
void invertLowerTriangular(const std::vector<double>& L, size_t n, std::vector<double>& T) {
    T.assign(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        T[i * n + i] = 1.0 / L[i * n + i];
        for (size_t j = 0; j < i; ++j) {
            double sum = 0;
            for (size_t k = j; k < i; ++k) sum -= L[i * n + k] * T[k * n + j];
            T[i * n + j] = sum / L[i * n + i];
        }
    }
}

// Build T such that ||T b||^2 = b^T G^+ b. Prefer Cholesky T=C^{-1}; else eigen factor.
std::vector<double> buildTransformT(const std::vector<double>& gram,
                                    size_t n,
                                    const std::vector<double>& pinv,
                                    size_t rank,
                                    size_t fallback) {
    std::vector<double> T(n * n, 0);
    if (n == 0) return T;
    if (fallback == 0 && rank == n) {
        std::vector<double> L = gram;
        if (choleskyLower(L, n)) {
            invertLowerTriangular(L, n, T);
            return T;
        }
    }
    // Eigen square-root factor of the already-computed pseudoinverse:
    // G^+ = U Λ^+ U^T => T = sqrt(Λ^+) U^T.
    const EigenResult eigen = symmetricJacobi(pinv, n);
    for (size_t k = 0; k < n; ++k) {
        const double value = eigen.values[k];
        if (value <= 1e-14) continue;
        const double scale = std::sqrt(value);
        for (size_t j = 0; j < n; ++j) {
            // Row k of T gets scale * eigenvector_k^T, but eigen.values are of pinv.
            // Using all positive eigendirections of pinv pads to n rows; equivalent
            // isometry on the range of G^+.
            T[k * n + j] = scale * eigen.vectors[j * n + k];
        }
    }
    return T;
}

void matvecSquare(const double* T, const double* b, size_t n, double* z) {
    for (size_t i = 0; i < n; ++i) {
        double sum = 0;
#pragma omp simd reduction(+ : sum)
        for (size_t j = 0; j < n; ++j) sum += T[i * n + j] * b[j];
        z[i] = sum;
    }
}

void fillMultilaterationB(const PivotMetadata& metadata,
                          const double* pivot_squared,
                          size_t active_count,
                          double* b) {
    const size_t n = active_count - 1;
    for (size_t i = 0; i < n; ++i) {
        const double vi2 = metadata.pairwise_squared[i + 1];
        b[i] = 0.5 * (pivot_squared[0] + vi2 - pivot_squared[i + 1]);
    }
}

std::vector<double> pseudoInverse(const std::vector<double>& gram,
                                  size_t n,
                                  std::vector<double>* eigenvalues,
                                  size_t* rank,
                                  double* condition,
                                  size_t* fallback) {
    const EigenResult eigen = symmetricJacobi(gram, n);
    if (eigenvalues) *eigenvalues = eigen.values;
    const double max_eigen = eigen.values.empty() ? 0 : std::max(0.0, eigen.values.front());
    // Rank revealing cutoff deliberately drops ill-conditioned directions.
    const double cutoff = std::max(1e-12, max_eigen * 1e-10);
    size_t retained = 0;
    double min_retained = std::numeric_limits<double>::infinity();
    for (double value : eigen.values) {
        if (value > cutoff) {
            ++retained;
            min_retained = std::min(min_retained, value);
        }
    }
    *rank = retained;
    *fallback = n - retained;
    *condition = retained == 0 ? std::numeric_limits<double>::infinity() : max_eigen / min_retained;

    std::vector<double> inverse(n * n, 0);
    for (size_t k = 0; k < n; ++k) {
        const double eigenvalue = eigen.values[k];
        if (eigenvalue <= cutoff) continue;
        const double inv = 1.0 / eigenvalue;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inverse[i * n + j] += eigen.vectors[i * n + k] * inv * eigen.vectors[j * n + k];
            }
        }
    }
    return inverse;
}

std::string joinDoubles(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::setprecision(10);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ';';
        out << values[i];
    }
    return out.str();
}

uint16_t floatToHalfBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if ((bits & 0x7fffffffu) == 0) return static_cast<uint16_t>(sign);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(1 - exp);
        uint32_t half_mant = mant >> (shift + 13);
        if ((mant >> (shift + 12)) & 1u) half_mant += 1;
        return static_cast<uint16_t>(sign | half_mant);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x1000u) half += 1;
    return static_cast<uint16_t>(half);
}

float halfBitsToFloat(uint16_t half) {
    const uint32_t sign = (static_cast<uint32_t>(half) & 0x8000u) << 16;
    uint32_t exp = (half >> 10) & 0x1fu;
    uint32_t mant = half & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float halfMaxAbsError(float decoded) {
    return std::max(1e-6f, std::abs(decoded) * (1.0f / 1024.0f) + 1e-6f);
}

}  // namespace

void PivotMetadata::clear() {
    *this = PivotMetadata{};
}

bool PivotMetadata::usable_projection(size_t active_count) const {
    const size_t n = count == 0 ? 0 : count - 1;
    return active_count == count && count >= 1 &&
           ((count == 1 && (gram_pinv.empty() || gram_pinv.size() == n * n)) ||
            gram_pinv.size() == n * n || U.size() == n * d);
}

bool PivotMetadata::has_transform(size_t active_count) const {
    return usable_projection(active_count) &&
           (active_count == 1 || transform_T.size() == (count - 1) * (count - 1));
}

bool PivotMetadata::has_U(size_t active_count) const {
    return active_count >= 1 && active_count <= count && count >= 1 && d != 0 &&
           (count == 1 || U.size() == (count - 1) * d);
}

bool PivotMetadata::usable_signature(size_t active_count) const {
    const bool has_local_geometry = has_transform(count) || has_U(count);
    if (active_count == 0 || active_count > count || signature_candidate_count == 0 ||
        (!external_projection_geometry && !has_local_geometry) ||
        (active_count != count && !external_projection_geometry && !has_U(count))) {
        return false;
    }
    const size_t expected = signature_candidate_count * count;
    if (signature_precision == SignaturePrecision::FLOAT32) {
        return signature_data.size() == expected;
    }
    if (signature_precision == SignaturePrecision::FLOAT16) {
        return signature_data_f16.size() == expected;
    }
    return signature_data_i8.size() == expected && signature_i8_scales.size() == count;
}

float PivotMetadata::signatureValue(size_t active_count, size_t dim, size_t candidate_index,
                                    float* max_abs_error) const {
    const size_t nc = signature_candidate_count;
    const size_t offset = dim * nc + candidate_index;
    if (signature_precision == SignaturePrecision::FLOAT32) {
        if (max_abs_error) *max_abs_error = 0.0f;
        return signature_data[offset];
    }
    if (signature_precision == SignaturePrecision::FLOAT16) {
        const float decoded = halfBitsToFloat(signature_data_f16[offset]);
        if (max_abs_error) *max_abs_error = halfMaxAbsError(decoded);
        return decoded;
    }
    const float scale = signature_i8_scales[dim];
    const float decoded = static_cast<float>(signature_data_i8[offset]) * scale;
    if (max_abs_error) *max_abs_error = 0.5f * scale;
    return decoded;
}

bool validPivotCount(size_t count) {
    return count >= 1 && count <= MULTIPIVOT_MAX_PIVOTS;
}

bool validMultipivotMethod(const std::string& method) {
    return method == "affine_fps" || method == "pca" || method == "weighted_pca" ||
           method == "irls_pca" || method == "greedy_prune";
}

bool isCentroidAnchoredMultipivotMethod(const std::string& method) {
    return method == "pca" || method == "weighted_pca" || method == "irls_pca" ||
           method == "greedy_prune";
}

PivotSelectionResult selectAffineFps(const float* candidates,
                                     const idx_t* source_ids,
                                     size_t candidate_count,
                                     size_t d,
                                     size_t requested_count,
                                     [[maybe_unused]] uint64_t seed,
                                     const float* forced_v0,
                                     idx_t forced_v0_id) {
    const auto start = std::chrono::steady_clock::now();
    PivotSelectionResult result;
    result.metadata.requested_count = requested_count;
    result.metadata.d = d;
    if (requested_count == 0) return result;
    if (forced_v0 == nullptr && candidate_count == 0) return result;

    auto sourceId = [&](size_t i) -> idx_t { return source_ids ? source_ids[i] : static_cast<idx_t>(i); };

    // Candidate indices chosen after v0. When forced_v0 is set, v0 is external
    // (IVF centroid); otherwise v0 is the first selected candidate.
    std::vector<size_t> selected_candidates;
    std::vector<bool> used(candidate_count, false);
    const float* p0_codes = nullptr;
    size_t count = 0;

    if (forced_v0 != nullptr) {
        p0_codes = forced_v0;
        const size_t from_candidates =
            std::min(candidate_count, requested_count > 0 ? requested_count - 1 : 0);
        count = 1 + from_candidates;
        result.metadata.count = count;
        selected_candidates.reserve(from_candidates);
    } else {
        const size_t count_from_pool = std::min(candidate_count, requested_count);
        count = count_from_pool;
        result.metadata.count = count;
        if (count == 0) return result;

        std::vector<double> mean(d, 0);
        for (size_t i = 0; i < candidate_count; ++i) {
            for (size_t j = 0; j < d; ++j) mean[j] += candidates[i * d + j];
        }
        for (double& value : mean) value /= candidate_count;

        size_t first = 0;
        double first_distance = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < candidate_count; ++i) {
            double distance = 0;
            for (size_t j = 0; j < d; ++j) {
                const double delta = static_cast<double>(candidates[i * d + j]) - mean[j];
                distance += delta * delta;
            }
            if (distance < first_distance ||
                (distance == first_distance && sourceId(i) < sourceId(first))) {
                first = i;
                first_distance = distance;
            }
        }
        selected_candidates.push_back(first);
        used[first] = true;
        p0_codes = candidates + first * d;
    }

    std::vector<double> basis;
    while (selected_candidates.size() + (forced_v0 != nullptr ? 1 : 0) < count) {
        // Add the newest independent affine direction using modified Gram-Schmidt.
        // Need at least one candidate beyond v0 before a direction exists.
        if (selected_candidates.size() >= (forced_v0 != nullptr ? 1 : 2)) {
            const float* newest = candidates + selected_candidates.back() * d;
            std::vector<double> direction(d);
            for (size_t j = 0; j < d; ++j) direction[j] = static_cast<double>(newest[j]) - p0_codes[j];
            for (size_t b = 0; b < basis.size() / d; ++b) {
                double dot = 0;
                for (size_t j = 0; j < d; ++j) dot += direction[j] * basis[b * d + j];
                for (size_t j = 0; j < d; ++j) direction[j] -= dot * basis[b * d + j];
            }
            double norm2 = std::inner_product(direction.begin(), direction.end(), direction.begin(), 0.0);
            if (norm2 > 1e-20) {
                const double inv_norm = 1.0 / std::sqrt(norm2);
                for (double value : direction) basis.push_back(value * inv_norm);
            }
        }

        size_t best = candidate_count;
        double best_residual = -1;
        for (size_t i = 0; i < candidate_count; ++i) {
            if (used[i]) continue;
            std::vector<double> delta(d);
            double residual = 0;
            for (size_t j = 0; j < d; ++j) {
                delta[j] = static_cast<double>(candidates[i * d + j]) - p0_codes[j];
                residual += delta[j] * delta[j];
            }
            for (size_t b = 0; b < basis.size() / d; ++b) {
                double dot = 0;
                for (size_t j = 0; j < d; ++j) dot += delta[j] * basis[b * d + j];
                residual -= dot * dot;
            }
            residual = std::max(0.0, residual);
            if (residual > best_residual ||
                (residual == best_residual && (best == candidate_count || sourceId(i) < sourceId(best)))) {
                best = i;
                best_residual = residual;
            }
        }
        if (best == candidate_count) break;
        selected_candidates.push_back(best);
        used[best] = true;
    }

    result.metadata.count = (forced_v0 != nullptr ? 1 : 0) + selected_candidates.size();
    result.metadata.source_ids.reserve(result.metadata.count);
    result.metadata.codes.reserve(result.metadata.count * d);
    if (forced_v0 != nullptr) {
        result.metadata.source_ids.push_back(forced_v0_id);
        result.metadata.codes.insert(result.metadata.codes.end(), forced_v0, forced_v0 + d);
    }
    for (size_t index : selected_candidates) {
        result.metadata.source_ids.push_back(sourceId(index));
        result.metadata.codes.insert(result.metadata.codes.end(),
                                     candidates + index * d, candidates + (index + 1) * d);
    }
    buildPivotGeometry(result.metadata);
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

namespace {

Eigen::VectorXd resolvePcaCenter(const float* candidates,
                                 size_t candidate_count,
                                 size_t d,
                                 const float* forced_mean,
                                 const float* sample_weights) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));
    if (forced_mean != nullptr) {
        for (size_t j = 0; j < d; ++j) {
            mean[static_cast<Eigen::Index>(j)] = static_cast<double>(forced_mean[j]);
        }
        return mean;
    }
    double weight_sum = 0.0;
    for (size_t i = 0; i < candidate_count; ++i) {
        const double w =
            sample_weights != nullptr ? static_cast<double>(sample_weights[i]) : 1.0;
        if (w <= 0.0) continue;
        weight_sum += w;
        for (size_t j = 0; j < d; ++j) {
            mean[static_cast<Eigen::Index>(j)] +=
                w * static_cast<double>(candidates[i * d + j]);
        }
    }
    if (weight_sum <= 0.0) {
        throw std::invalid_argument("PCA center requires positive total sample weight");
    }
    mean /= weight_sum;
    return mean;
}

// Fixed-center weighted PCA: top-m eigenvectors of Σ w_i (x_i-c)(x_i-c)^T / Σ w.
void weightedPcaFixedCenterDirections(const float* candidates,
                                      size_t candidate_count,
                                      size_t d,
                                      const Eigen::VectorXd& center,
                                      const float* sample_weights,
                                      size_t m,
                                      std::vector<Eigen::VectorXd>& directions,
                                      std::vector<double>* eigenvalues_out) {
    directions.clear();
    const Eigen::Index dim = static_cast<Eigen::Index>(d);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(dim, dim);
    double weight_sum = 0.0;
    for (size_t i = 0; i < candidate_count; ++i) {
        const double w =
            sample_weights != nullptr ? static_cast<double>(sample_weights[i]) : 1.0;
        if (w <= 0.0) continue;
        weight_sum += w;
        Eigen::VectorXd delta(dim);
        for (size_t j = 0; j < d; ++j) {
            delta[static_cast<Eigen::Index>(j)] =
                static_cast<double>(candidates[i * d + j]) - center[static_cast<Eigen::Index>(j)];
        }
        cov.noalias() += w * (delta * delta.transpose());
    }
    if (weight_sum <= 0.0) {
        throw std::invalid_argument("weighted PCA requires positive total sample weight");
    }
    cov /= weight_sum;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("PCA eigendecomposition failed");
    }
    if (eigenvalues_out != nullptr) {
        eigenvalues_out->resize(static_cast<size_t>(dim));
        for (Eigen::Index k = 0; k < dim; ++k) {
            (*eigenvalues_out)[static_cast<size_t>(k)] =
                std::max(0.0, solver.eigenvalues()[dim - 1 - k]);
        }
    }

    constexpr double kMinEigenvalue = 1e-12;
    directions.reserve(m);
    const size_t available = std::min(m, d);
    for (size_t k = 0; k < available; ++k) {
        const Eigen::Index eigen_index = dim - 1 - static_cast<Eigen::Index>(k);
        if (solver.eigenvalues()[eigen_index] <= kMinEigenvalue) break;
        Eigen::VectorXd u = solver.eigenvectors().col(eigen_index);
        const double norm = u.norm();
        if (norm <= 1e-12) continue;
        u /= norm;
        for (Eigen::Index j = 0; j < dim; ++j) {
            if (std::abs(u[j]) > 1e-12) {
                if (u[j] < 0) u = -u;
                break;
            }
        }
        directions.push_back(std::move(u));
    }
    for (size_t axis = 0; directions.size() < m && axis < d; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(dim);
        u[static_cast<Eigen::Index>(axis)] = 1.0;
        for (const Eigen::VectorXd& existing : directions) {
            u -= existing.dot(u) * existing;
        }
        const double norm = u.norm();
        if (norm <= 1e-12) continue;
        u /= norm;
        directions.push_back(std::move(u));
    }
    if (directions.size() < m) {
        throw std::runtime_error("Unable to build enough PCA directions");
    }
}

void fillPcaPivotCodes(PivotMetadata& metadata,
                       const Eigen::VectorXd& center,
                       const std::vector<Eigen::VectorXd>& directions,
                       idx_t forced_mean_id,
                       bool forced) {
    const size_t d = metadata.d;
    const size_t r = directions.size();
    metadata.count = r + 1;
    metadata.source_ids.clear();
    metadata.source_ids.reserve(metadata.count);
    metadata.codes.resize(metadata.count * d);
    metadata.source_ids.push_back(forced ? forced_mean_id : static_cast<idx_t>(-1));
    for (size_t j = 0; j < d; ++j) {
        metadata.codes[j] = static_cast<float>(center[static_cast<Eigen::Index>(j)]);
    }
    for (size_t i = 0; i < r; ++i) {
        metadata.source_ids.push_back(-static_cast<idx_t>(i) - 2);
        const Eigen::VectorXd point = center + directions[i];
        for (size_t j = 0; j < d; ++j) {
            metadata.codes[(i + 1) * d + j] =
                static_cast<float>(point[static_cast<Eigen::Index>(j)]);
        }
    }
}

}  // namespace

PivotSelectionResult selectWeightedPcaPivots(const float* candidates,
                                             size_t candidate_count,
                                             size_t d,
                                             size_t requested_count,
                                             const float* forced_mean,
                                             idx_t forced_mean_id,
                                             const float* sample_weights) {
    const auto start = std::chrono::steady_clock::now();
    PivotSelectionResult result;
    result.metadata.requested_count = requested_count;
    result.metadata.d = d;
    if (requested_count == 0 || candidate_count == 0 || d == 0) return result;
    if (!validPivotCount(requested_count)) {
        throw std::invalid_argument("PCA pivot_count must be in 1..MULTIPIVOT_MAX_PIVOTS");
    }

    const Eigen::VectorXd center =
        resolvePcaCenter(candidates, candidate_count, d, forced_mean, sample_weights);
    const size_t r = requested_count - 1;
    std::vector<Eigen::VectorXd> directions;
    if (r == 0) {
        fillPcaPivotCodes(result.metadata, center, directions, forced_mean_id,
                          forced_mean != nullptr);
        buildPivotGeometry(result.metadata);
        buildPcaProjectionU(result.metadata);
        result.elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    }

    weightedPcaFixedCenterDirections(candidates, candidate_count, d, center, sample_weights, r,
                                     directions, &result.pca_cov_eigenvalues);
    fillPcaPivotCodes(result.metadata, center, directions, forced_mean_id, forced_mean != nullptr);
    buildPivotGeometry(result.metadata);
    buildPcaProjectionU(result.metadata);
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

PivotSelectionResult selectPcaPivots(const float* candidates,
                                     size_t candidate_count,
                                     size_t d,
                                     size_t requested_count,
                                     const float* forced_mean,
                                     idx_t forced_mean_id) {
    return selectWeightedPcaPivots(candidates, candidate_count, d, requested_count, forced_mean,
                                   forced_mean_id, nullptr);
}

PivotSelectionResult selectIrlsPcaPivots(const float* candidates,
                                         size_t candidate_count,
                                         size_t d,
                                         size_t requested_count,
                                         const float* forced_mean,
                                         idx_t forced_mean_id,
                                         size_t max_iter,
                                         float residual_floor,
                                         const float* alpha) {
    const auto start = std::chrono::steady_clock::now();
    PivotSelectionResult result;
    result.metadata.requested_count = requested_count;
    result.metadata.d = d;
    if (requested_count == 0 || candidate_count == 0 || d == 0) return result;
    if (!validPivotCount(requested_count)) {
        throw std::invalid_argument("IRLS PCA pivot_count must be in 1..MULTIPIVOT_MAX_PIVOTS");
    }
    if (max_iter == 0) {
        throw std::invalid_argument("irls_max_iter must be >= 1");
    }
    if (!(residual_floor > 0.0f) || !std::isfinite(residual_floor)) {
        throw std::invalid_argument("irls_residual_floor must be a positive finite float");
    }

    const Eigen::VectorXd center =
        resolvePcaCenter(candidates, candidate_count, d, forced_mean, alpha);
    const size_t r = requested_count - 1;
    std::vector<Eigen::VectorXd> directions;
    if (r == 0) {
        fillPcaPivotCodes(result.metadata, center, directions, forced_mean_id,
                          forced_mean != nullptr);
        buildPivotGeometry(result.metadata);
        buildPcaProjectionU(result.metadata);
        result.elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    }

    std::vector<float> weights(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
        const float a = alpha != nullptr ? alpha[i] : 1.0f;
        weights[i] = a > 0.0f ? a : 0.0f;
    }
    const double eps = static_cast<double>(residual_floor);
    result.irls_objective_history.reserve(max_iter);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        weightedPcaFixedCenterDirections(candidates, candidate_count, d, center, weights.data(), r,
                                         directions,
                                         iter + 1 == max_iter ? &result.pca_cov_eigenvalues
                                                              : nullptr);

        // Objective Σ α_i ||(I-UU^T)(x_i-c)|| and weight update w_i = α_i / max(h_i, ε).
        double objective = 0.0;
        for (size_t i = 0; i < candidate_count; ++i) {
            const double a =
                alpha != nullptr ? static_cast<double>(alpha[i]) : 1.0;
            if (a <= 0.0) {
                weights[i] = 0.0f;
                continue;
            }
            Eigen::VectorXd delta(static_cast<Eigen::Index>(d));
            double delta_norm2 = 0.0;
            for (size_t j = 0; j < d; ++j) {
                const double v =
                    static_cast<double>(candidates[i * d + j]) - center[static_cast<Eigen::Index>(j)];
                delta[static_cast<Eigen::Index>(j)] = v;
                delta_norm2 += v * v;
            }
            double proj_norm2 = 0.0;
            for (size_t k = 0; k < r; ++k) {
                const double coord = directions[k].dot(delta);
                proj_norm2 += coord * coord;
            }
            const double h = std::sqrt(std::max(0.0, delta_norm2 - proj_norm2));
            objective += a * h;
            weights[i] = static_cast<float>(a / std::max(h, eps));
        }
        result.irls_objective_history.push_back(objective);
    }

    fillPcaPivotCodes(result.metadata, center, directions, forced_mean_id, forced_mean != nullptr);
    buildPivotGeometry(result.metadata);
    buildPcaProjectionU(result.metadata);
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

namespace {

struct GreedyPair {
    const float* q_codes = nullptr;  // length d (candidate or real query)
    size_t x = 0;
    double true_dist2 = 0;
    double tau2 = 0;
    double lb2 = 0;
    double hq = 0;
    double hx = 0;
};

Eigen::VectorXd pointMinusCenter(const float* codes,
                                 size_t d,
                                 const Eigen::VectorXd& center) {
    Eigen::VectorXd a(static_cast<Eigen::Index>(d));
    for (size_t j = 0; j < d; ++j) {
        a[static_cast<Eigen::Index>(j)] =
            static_cast<double>(codes[j]) - center[static_cast<Eigen::Index>(j)];
    }
    return a;
}

Eigen::VectorXd pointMinusCenter(const float* candidates,
                                 size_t index,
                                 size_t d,
                                 const Eigen::VectorXd& center) {
    return pointMinusCenter(candidates + index * d, d, center);
}

Eigen::VectorXd residualToSubspace(const Eigen::VectorXd& a_y,
                                   const std::vector<Eigen::VectorXd>& directions) {
    Eigen::VectorXd e = a_y;
    for (const Eigen::VectorXd& u : directions) {
        e -= u.dot(e) * u;
    }
    return e;
}

void orthonormalizeAgainst(Eigen::VectorXd& v,
                           const std::vector<Eigen::VectorXd>& basis) {
    for (const Eigen::VectorXd& u : basis) {
        v -= u.dot(v) * u;
    }
    const double n = v.norm();
    if (n > 1e-12) v /= n;
    else v.setZero();
}

Eigen::VectorXd powerIterateProxy(const std::vector<Eigen::VectorXd>& residuals,
                                  const std::vector<Eigen::VectorXd>& forbid,
                                  size_t d,
                                  size_t iters,
                                  std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd v(static_cast<Eigen::Index>(d));
    for (size_t j = 0; j < d; ++j) v[static_cast<Eigen::Index>(j)] = normal(rng);
    orthonormalizeAgainst(v, forbid);
    if (v.norm() <= 1e-12) return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));

    for (size_t it = 0; it < iters; ++it) {
        Eigen::VectorXd Mv = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));
        for (const Eigen::VectorXd& r : residuals) {
            Mv += r.dot(v) * r;
        }
        orthonormalizeAgainst(Mv, forbid);
        if (Mv.norm() <= 1e-12) {
            return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));
        }
        v = Mv;
    }
    // Deterministic sign.
    for (Eigen::Index j = 0; j < v.size(); ++j) {
        if (std::abs(v[j]) > 1e-12) {
            if (v[j] < 0) v = -v;
            break;
        }
    }
    return v;
}

double lbAfterDirection(double lb2,
                        double hq,
                        double hx,
                        double sq,
                        double sx) {
    const double hq2 = std::max(0.0, hq * hq - sq * sq);
    const double hx2 = std::max(0.0, hx * hx - sx * sx);
    const double hq_new = std::sqrt(hq2);
    const double hx_new = std::sqrt(hx2);
    const double delta =
        (sq - sx) * (sq - sx) + (hq_new - hx_new) * (hq_new - hx_new) - (hq - hx) * (hq - hx);
    return std::max(0.0, lb2 + delta);
}

}  // namespace

PivotSelectionResult selectGreedyPrunePivots(const float* candidates,
                                             size_t candidate_count,
                                             size_t d,
                                             size_t requested_count,
                                             const float* forced_mean,
                                             idx_t forced_mean_id,
                                             uint64_t seed,
                                             const GreedyPruneConfig& config,
                                             const float* train_queries,
                                             size_t train_query_count) {
    const auto start = std::chrono::steady_clock::now();
    PivotSelectionResult result;
    result.metadata.requested_count = requested_count;
    result.metadata.d = d;
    if (requested_count == 0 || candidate_count == 0 || d == 0) return result;
    if (!validPivotCount(requested_count)) {
        throw std::invalid_argument("greedy_prune pivot_count must be in 1..MULTIPIVOT_MAX_PIVOTS");
    }
    // Too small to form 1-NN pairs: fall back to fixed-center PCA.
    if (candidate_count < 2) {
        return selectPcaPivots(candidates, candidate_count, d, requested_count, forced_mean,
                               forced_mean_id);
    }

    const Eigen::VectorXd center =
        resolvePcaCenter(candidates, candidate_count, d, forced_mean, nullptr);
    const size_t m = requested_count - 1;
    std::vector<Eigen::VectorXd> directions;
    if (m == 0) {
        fillPcaPivotCodes(result.metadata, center, directions, forced_mean_id,
                          forced_mean != nullptr);
        buildPivotGeometry(result.metadata);
        buildPcaProjectionU(result.metadata);
        result.elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    }

    std::mt19937_64 rng(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed);

    // Training queries: prefer real queries when provided; else sample from the pool.
    const bool use_real = train_queries != nullptr && train_query_count > 0;
    std::vector<const float*> query_ptrs;
    if (use_real) {
        query_ptrs.reserve(train_query_count);
        for (size_t i = 0; i < train_query_count; ++i) {
            query_ptrs.push_back(train_queries + i * d);
        }
        // Cap extremely large assignments with pseudo_nq as a soft ceiling (0 disables).
        if (config.pseudo_nq > 0 && query_ptrs.size() > config.pseudo_nq) {
            std::shuffle(query_ptrs.begin(), query_ptrs.end(), rng);
            query_ptrs.resize(config.pseudo_nq);
        }
    } else {
        std::vector<size_t> order(candidate_count);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        const size_t nq = std::min(config.pseudo_nq, candidate_count);
        query_ptrs.reserve(nq);
        for (size_t i = 0; i < nq; ++i) {
            query_ptrs.push_back(candidates + order[i] * d);
        }
    }
    const size_t nq = query_ptrs.size();

    std::vector<double> tau2(nq, 0.0);
    std::vector<size_t> nn_ids(nq, 0);
    for (size_t qi = 0; qi < nq; ++qi) {
        const float* q = query_ptrs[qi];
        double best = std::numeric_limits<double>::infinity();
        size_t best_j = 0;
        for (size_t j = 0; j < candidate_count; ++j) {
            // For pool-sampled queries equal to candidate j, skip self-match.
            if (!use_real && q == candidates + j * d) continue;
            const double dist2 = squaredL2(q, candidates + j * d, d);
            if (dist2 < best) {
                best = dist2;
                best_j = j;
            }
        }
        tau2[qi] = best;
        nn_ids[qi] = best_j;
    }

    // Negatives with ||q-x||^2 > τ^2; prefer near-boundary.
    std::vector<GreedyPair> pairs;
    pairs.reserve(nq * std::min(config.negatives_per_q, candidate_count));
    const double alpha = std::max(0.0f, config.boundary_alpha);
    for (size_t qi = 0; qi < nq; ++qi) {
        const float* q = query_ptrs[qi];
        const double t2 = tau2[qi];
        if (!(t2 > 0.0) || !std::isfinite(t2)) continue;
        const double upper = t2 * (1.0 + alpha) * (1.0 + alpha);

        std::vector<size_t> boundary;
        std::vector<size_t> far;
        boundary.reserve(candidate_count);
        far.reserve(candidate_count);
        for (size_t j = 0; j < candidate_count; ++j) {
            if (j == nn_ids[qi]) continue;
            if (!use_real && q == candidates + j * d) continue;
            const double dist2 = squaredL2(q, candidates + j * d, d);
            if (!(dist2 > t2)) continue;
            if (dist2 <= upper) boundary.push_back(j);
            else far.push_back(j);
        }
        std::shuffle(boundary.begin(), boundary.end(), rng);
        std::shuffle(far.begin(), far.end(), rng);

        size_t taken = 0;
        auto push_neg = [&](size_t x) {
            if (taken >= config.negatives_per_q) return;
            const double dist2 = squaredL2(q, candidates + x * d, d);
            const Eigen::VectorXd aq = pointMinusCenter(q, d, center);
            const Eigen::VectorXd ax = pointMinusCenter(candidates, x, d, center);
            GreedyPair pair;
            pair.q_codes = q;
            pair.x = x;
            pair.true_dist2 = dist2;
            pair.tau2 = t2;
            pair.hq = aq.norm();
            pair.hx = ax.norm();
            pair.lb2 = (pair.hq - pair.hx) * (pair.hq - pair.hx);
            pairs.push_back(pair);
            ++taken;
        };
        for (size_t x : boundary) {
            if (taken >= config.negatives_per_q) break;
            push_neg(x);
        }
        for (size_t x : far) {
            if (taken >= config.negatives_per_q) break;
            push_neg(x);
        }
    }

    if (pairs.empty()) {
        // No trainable hard pairs: fixed-center PCA fallback.
        return selectPcaPivots(candidates, candidate_count, d, requested_count, forced_mean,
                               forced_mean_id);
    }

    auto padWithPca = [&](size_t need) {
        if (need == 0) return;
        std::vector<Eigen::VectorXd> pca_dirs;
        weightedPcaFixedCenterDirections(candidates, candidate_count, d, center, nullptr, need + directions.size(),
                                         pca_dirs, directions.empty() ? &result.pca_cov_eigenvalues
                                                                      : nullptr);
        for (Eigen::VectorXd& u : pca_dirs) {
            orthonormalizeAgainst(u, directions);
            if (u.norm() <= 1e-12) continue;
            directions.push_back(std::move(u));
            if (directions.size() >= m) break;
        }
        // Identity completion if still short.
        for (size_t axis = 0; directions.size() < m && axis < d; ++axis) {
            Eigen::VectorXd u = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));
            u[static_cast<Eigen::Index>(axis)] = 1.0;
            orthonormalizeAgainst(u, directions);
            if (u.norm() <= 1e-12) continue;
            directions.push_back(std::move(u));
        }
    };

    for (size_t t = 0; t < m; ++t) {
        std::vector<size_t> hard_idx;
        hard_idx.reserve(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) {
            const GreedyPair& p = pairs[i];
            if (p.lb2 <= p.tau2 && p.tau2 < p.true_dist2) hard_idx.push_back(i);
        }
        if (hard_idx.empty()) {
            padWithPca(m - directions.size());
            break;
        }

        std::vector<Eigen::VectorXd> residuals;
        residuals.reserve(hard_idx.size());
        std::vector<std::pair<double, size_t>> residual_norms;
        residual_norms.reserve(hard_idx.size());
        for (size_t hi = 0; hi < hard_idx.size(); ++hi) {
            const GreedyPair& p = pairs[hard_idx[hi]];
            const Eigen::VectorXd eq =
                residualToSubspace(pointMinusCenter(p.q_codes, d, center), directions);
            const Eigen::VectorXd ex =
                residualToSubspace(pointMinusCenter(candidates, p.x, d, center), directions);
            Eigen::VectorXd r = eq - ex;
            const double rn = r.norm();
            residual_norms.emplace_back(rn, residuals.size());
            residuals.push_back(std::move(r));
        }

        std::vector<Eigen::VectorXd> forbid = directions;
        std::vector<Eigen::VectorXd> candidates_v;
        candidates_v.reserve(config.proxy_candidates + config.pair_direction_candidates);

        for (size_t cidx = 0; cidx < config.proxy_candidates; ++cidx) {
            Eigen::VectorXd v =
                powerIterateProxy(residuals, forbid, d, config.power_iters, rng);
            if (v.norm() <= 1e-12) break;
            candidates_v.push_back(v);
            forbid.push_back(v);
        }

        std::sort(residual_norms.begin(), residual_norms.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t k = 0; k < residual_norms.size() &&
                           candidates_v.size() < config.proxy_candidates + config.pair_direction_candidates;
             ++k) {
            if (residual_norms[k].first <= 1e-12) continue;
            Eigen::VectorXd v = residuals[residual_norms[k].second];
            orthonormalizeAgainst(v, directions);
            for (const Eigen::VectorXd& existing : candidates_v) {
                v -= existing.dot(v) * existing;
            }
            const double n = v.norm();
            if (n <= 1e-12) continue;
            v /= n;
            candidates_v.push_back(std::move(v));
        }

        if (candidates_v.empty()) {
            padWithPca(m - directions.size());
            break;
        }

        double best_gain = -1.0;
        size_t best_idx = 0;
        for (size_t ci = 0; ci < candidates_v.size(); ++ci) {
            const Eigen::VectorXd& v = candidates_v[ci];
            double gain = 0.0;
            for (size_t hi : hard_idx) {
                const GreedyPair& p = pairs[hi];
                const Eigen::VectorXd aq = pointMinusCenter(p.q_codes, d, center);
                const Eigen::VectorXd ax = pointMinusCenter(candidates, p.x, d, center);
                const double sq = v.dot(aq);
                const double sx = v.dot(ax);
                const double lb_new = lbAfterDirection(p.lb2, p.hq, p.hx, sq, sx);
                if (p.lb2 <= p.tau2 && p.tau2 < lb_new) gain += 1.0;
            }
            if (gain > best_gain) {
                best_gain = gain;
                best_idx = ci;
            }
        }

        if (best_gain < config.min_gain) {
            padWithPca(m - directions.size());
            break;
        }

        const Eigen::VectorXd chosen = candidates_v[best_idx];
        // Commit direction and refresh pair residuals / LB (build-time only).
        for (GreedyPair& p : pairs) {
            const Eigen::VectorXd aq = pointMinusCenter(p.q_codes, d, center);
            const Eigen::VectorXd ax = pointMinusCenter(candidates, p.x, d, center);
            const double sq = chosen.dot(aq);
            const double sx = chosen.dot(ax);
            p.lb2 = lbAfterDirection(p.lb2, p.hq, p.hx, sq, sx);
            p.hq = std::sqrt(std::max(0.0, p.hq * p.hq - sq * sq));
            p.hx = std::sqrt(std::max(0.0, p.hx * p.hx - sx * sx));
        }
        directions.push_back(chosen);
        result.greedy_gain_history.push_back(best_gain);
    }

    if (directions.size() < m) padWithPca(m - directions.size());
    if (directions.size() > m) directions.resize(m);

    fillPcaPivotCodes(result.metadata, center, directions, forced_mean_id, forced_mean != nullptr);
    buildPivotGeometry(result.metadata);
    buildPcaProjectionU(result.metadata);
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

void buildPivotGeometry(PivotMetadata& metadata) {
    const size_t p_count = metadata.count;
    metadata.pairwise_squared.assign(p_count * p_count, 0);
    for (size_t i = 0; i < p_count; ++i) {
        for (size_t j = i + 1; j < p_count; ++j) {
            const double distance = squaredL2(metadata.codes.data() + i * metadata.d,
                                              metadata.codes.data() + j * metadata.d,
                                              metadata.d);
            metadata.pairwise_squared[i * p_count + j] = distance;
            metadata.pairwise_squared[j * p_count + i] = distance;
        }
    }

    metadata.gram_pinv.clear();
    metadata.gram_eigenvalues.clear();
    metadata.transform_T.clear();
    metadata.U.clear();
    metadata.signature_data.clear();
    metadata.signature_candidate_count = 0;

    if (p_count == 1) {
        metadata.rank = 0;
        metadata.condition = 1.0;
        metadata.fallback_count = 0;
        return;
    }
    const size_t n = p_count - 1;
    std::vector<double> gram = gramForPrefix(metadata, p_count);
    metadata.gram_pinv =
        pseudoInverse(gram, n, &metadata.gram_eigenvalues, &metadata.rank,
                      &metadata.condition, &metadata.fallback_count);
    metadata.transform_T =
        buildTransformT(gram, n, metadata.gram_pinv, metadata.rank, metadata.fallback_count);
}

void buildPcaProjectionU(PivotMetadata& metadata) {
    const size_t p = metadata.count;
    if (p == 0 || metadata.d == 0) {
        throw std::invalid_argument("Cannot build PCA projection U without pivots and dimension");
    }
    const size_t n = p - 1;
    metadata.U.clear();
    if (n == 0) {
        metadata.transform_T.clear();
        return;
    }
    if (metadata.codes.size() != p * metadata.d ||
        metadata.transform_T.size() != n * n) {
        throw std::runtime_error("Invalid PCA geometry while building fused projection U");
    }

    metadata.U.assign(n * metadata.d, 0.0f);
    const float* center = metadata.codes.data();
    for (size_t row = 0; row < n; ++row) {
        for (size_t axis = 0; axis < metadata.d; ++axis) {
            double value = 0.0;
            for (size_t direction = 0; direction < n; ++direction) {
                const double component =
                    static_cast<double>(metadata.codes[(direction + 1) * metadata.d + axis]) -
                    static_cast<double>(center[axis]);
                value += metadata.transform_T[row * n + direction] * component;
            }
            metadata.U[row * metadata.d + axis] = static_cast<float>(value);
        }
    }

    // PCA-family search and signature construction now consume U directly.
    metadata.transform_T.clear();
    metadata.transform_T.shrink_to_fit();
}

void computeCandidatePivotDistances(const float* candidates,
                                    size_t candidate_count,
                                    size_t d,
                                    PivotMetadata& metadata) {
    metadata.candidate_centroid_squared.resize(candidate_count);
    if (metadata.has_U(metadata.count)) {
        // Projection/PCA search only consumes the progressive signature and the
        // squared distance to the list centroid.  Materialising every
        // candidate-to-virtual-pivot distance duplicates the same information
        // and costs 8 * candidate_count * pivot_count bytes.  That temporary is
        // especially harmful for large indexes (256 GB for BIGANN1B/P=32).
        metadata.candidate_pivot_squared.clear();
        metadata.candidate_pivot_squared.shrink_to_fit();
        metadata.signature_candidate_count = candidate_count;
        if (metadata.count == 0 || candidate_count == 0) return;

        const size_t n = metadata.count - 1;
        metadata.signature_data.assign(candidate_count * metadata.count, 0.0f);
        const float* center = metadata.codes.data();
        std::vector<float> centered(d);
        for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
            const float* x = candidates + candidate * d;
            double center_distance_squared = 0.0;
            for (size_t axis = 0; axis < d; ++axis) {
                centered[axis] = x[axis] - center[axis];
                center_distance_squared +=
                    static_cast<double>(centered[axis]) * centered[axis];
            }
            metadata.candidate_centroid_squared[candidate] =
                static_cast<float>(center_distance_squared);
            float z_norm = 0.0f;
            for (size_t row = 0; row < n; ++row) {
                float projection = 0.0f;
#pragma omp simd reduction(+ : projection)
                for (size_t axis = 0; axis < d; ++axis) {
                    projection += centered[axis] * metadata.U[row * d + axis];
                }
                metadata.signature_data[row * candidate_count + candidate] = projection;
                z_norm += projection * projection;
            }
            metadata.signature_data[n * candidate_count + candidate] = static_cast<float>(
                std::sqrt(std::max(0.0, center_distance_squared - z_norm)));
        }
    } else {
        metadata.candidate_pivot_squared.resize(candidate_count * metadata.count);
        for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
            for (size_t pivot = 0; pivot < metadata.count; ++pivot) {
                metadata.candidate_pivot_squared[candidate * metadata.count + pivot] =
                    squaredL2(candidates + candidate * d,
                              metadata.codes.data() + pivot * d, d);
            }
            metadata.candidate_centroid_squared[candidate] =
                static_cast<float>(metadata.candidate_pivot_squared[candidate * metadata.count]);
        }
        computeCandidateSignatures(metadata, candidate_count);
    }
}

void computeCandidateSignatures(PivotMetadata& metadata,
                                size_t candidate_count) {
    metadata.signature_candidate_count = candidate_count;
    if (metadata.count == 0 || candidate_count == 0) return;
    const size_t p = metadata.count;
    const size_t n = p - 1;
    // Dim-major SoA: data[dim * candidate_count + cand], last dim = sqrt(rho).
    metadata.signature_data.assign(candidate_count * p, 0.0f);
    std::vector<double> b(MULTIPIVOT_MAX_PIVOTS);
    std::vector<double> z(MULTIPIVOT_MAX_PIVOTS);
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const double* pivot_squared =
            metadata.candidate_pivot_squared.data() + candidate * metadata.count;
        float* soa = metadata.signature_data.data();
        if (p == 1) {
            soa[candidate] =
                static_cast<float>(std::sqrt(std::max(0.0, pivot_squared[0])));
            continue;
        }
        fillMultilaterationB(metadata, pivot_squared, p, b.data());
        matvecSquare(metadata.transform_T.data(), b.data(), n, z.data());
        double z_norm = 0;
        for (size_t i = 0; i < n; ++i) {
            soa[i * candidate_count + candidate] = static_cast<float>(z[i]);
            z_norm += z[i] * z[i];
        }
        soa[n * candidate_count + candidate] =
            static_cast<float>(std::sqrt(std::max(0.0, pivot_squared[0] - z_norm)));
    }
}

void compressCandidateSignatures(PivotMetadata& metadata, SignaturePrecision precision) {
    metadata.candidate_prefix_sqrt_rho.clear();
    metadata.candidate_prefix_residual_squared.clear();
    metadata.candidate_prefix_length = 0;
    metadata.signature_data_f16.clear();
    metadata.signature_data_i8.clear();
    metadata.signature_i8_scales.clear();
    metadata.signature_precision = SignaturePrecision::FLOAT32;
    if (precision == SignaturePrecision::FLOAT32 || metadata.signature_data.empty() ||
        metadata.signature_candidate_count == 0) {
        return;
    }
    const size_t nc = metadata.signature_candidate_count;
    if (precision == SignaturePrecision::FLOAT16) {
        metadata.signature_data_f16.resize(metadata.signature_data.size());
        for (size_t i = 0; i < metadata.signature_data.size(); ++i) {
            metadata.signature_data_f16[i] = floatToHalfBits(metadata.signature_data[i]);
        }
        metadata.signature_data.clear();
        metadata.signature_data.shrink_to_fit();
        metadata.signature_precision = SignaturePrecision::FLOAT16;
        return;
    }

    // FLOAT8: one per-dimension scale; SoA layout is unchanged.
    metadata.signature_data_i8.resize(metadata.signature_data.size());
    metadata.signature_i8_scales.assign(metadata.count, 0.0f);
    for (size_t dim = 0; dim < metadata.count; ++dim) {
        float max_abs = 0.0f;
        for (size_t c = 0; c < nc; ++c) {
            max_abs = std::max(max_abs, std::abs(metadata.signature_data[dim * nc + c]));
        }
        const float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        metadata.signature_i8_scales[dim] = scale;
        const float inv = 1.0f / scale;
        for (size_t c = 0; c < nc; ++c) {
            const float q = std::round(metadata.signature_data[dim * nc + c] * inv);
            const int clipped = static_cast<int>(std::max(-127.0f, std::min(127.0f, q)));
            metadata.signature_data_i8[dim * nc + c] = static_cast<int8_t>(clipped);
        }
    }
    metadata.signature_data.clear();
    metadata.signature_data.shrink_to_fit();
    metadata.signature_precision = SignaturePrecision::FLOAT8;
}

void prepareCandidatePrefixResiduals(PivotMetadata& metadata, size_t prefix_length) {
    if (prefix_length == 0 || metadata.count < 2 ||
        metadata.signature_precision != SignaturePrecision::FLOAT32 ||
        metadata.signature_data.empty() || metadata.signature_candidate_count == 0 ||
        (metadata.candidate_centroid_squared.size() != metadata.signature_candidate_count &&
         metadata.candidate_pivot_squared.size() !=
             metadata.signature_candidate_count * metadata.count)) {
        metadata.candidate_prefix_sqrt_rho.clear();
        metadata.candidate_prefix_residual_squared.clear();
        metadata.candidate_prefix_length = 0;
        return;
    }

    const size_t n = metadata.count - 1;
    const size_t prefix_count = std::min(prefix_length, n);
    const size_t nc = metadata.signature_candidate_count;
    const bool extend_existing =
        metadata.candidate_prefix_sqrt_rho.size() == nc &&
        metadata.candidate_prefix_residual_squared.size() == nc &&
        metadata.candidate_prefix_length <= prefix_count;
    const size_t begin_dim = extend_existing ? metadata.candidate_prefix_length : 0;
    if (extend_existing && begin_dim == prefix_count) return;
    if (!extend_existing) {
        metadata.candidate_prefix_sqrt_rho.resize(nc);
        metadata.candidate_prefix_residual_squared.resize(nc);
    }
#pragma omp parallel for
    for (size_t candidate = 0; candidate < nc; ++candidate) {
        double remaining = extend_existing
            ? metadata.candidate_prefix_residual_squared[candidate]
            : metadata.candidate_centroid_squared.empty()
                ? metadata.candidate_pivot_squared[
                      candidate * metadata.count]
                : metadata.candidate_centroid_squared[candidate];
        for (size_t dim = begin_dim; dim < prefix_count; ++dim) {
            const float value = metadata.signature_data[dim * nc + candidate];
            remaining = std::max(0.0, remaining - static_cast<double>(value) * value);
        }
        metadata.candidate_prefix_residual_squared[candidate] = remaining;
        metadata.candidate_prefix_sqrt_rho[candidate] =
            static_cast<float>(std::sqrt(remaining));
    }
    metadata.candidate_prefix_length = prefix_count;
}

void computeQueryPivotDistances(const float* query,
                                size_t d,
                                const PivotMetadata& metadata,
                                size_t active_count,
                                double* squared_distances) {
    for (size_t pivot = 0; pivot < active_count; ++pivot) {
        squared_distances[pivot] = squaredL2(query, metadata.codes.data() + pivot * d, d);
    }
}

bool computeQuerySignature(const PivotMetadata& metadata,
                           const double* query_pivot_squared,
                           size_t active_count,
                           float* z_query,
                           float* sqrt_rho_query) {
    if (!metadata.has_transform(active_count)) return false;
    if (active_count == 1) {
        *sqrt_rho_query = static_cast<float>(std::sqrt(std::max(0.0, query_pivot_squared[0])));
        return std::isfinite(*sqrt_rho_query);
    }
    const size_t n = active_count - 1;
    std::array<double, MULTIPIVOT_MAX_PIVOTS> b{};
    std::array<double, MULTIPIVOT_MAX_PIVOTS> z{};
    fillMultilaterationB(metadata, query_pivot_squared, active_count, b.data());
    matvecSquare(metadata.transform_T.data(), b.data(), n, z.data());
    double z_norm = 0;
    for (size_t i = 0; i < n; ++i) {
        z_query[i] = static_cast<float>(z[i]);
        z_norm += z[i] * z[i];
    }
    *sqrt_rho_query = static_cast<float>(std::sqrt(std::max(0.0, query_pivot_squared[0] - z_norm)));
    return std::isfinite(*sqrt_rho_query);
}

bool computePcaQuerySignature(const float* query,
                              size_t d,
                              const PivotMetadata& metadata,
                              size_t active_count,
                              double* query_pivot_squared,
                              float* z_query,
                              float* sqrt_rho_query,
                              float* centered_query) {
    if (query == nullptr || query_pivot_squared == nullptr || z_query == nullptr ||
        sqrt_rho_query == nullptr || centered_query == nullptr || d != metadata.d ||
        active_count == 0 || active_count > metadata.count || !metadata.has_U(active_count)) {
        return false;
    }

    const float* center = metadata.codes.data();
    float center_distance_squared = 0.0f;
#pragma omp simd reduction(+ : center_distance_squared)
    for (size_t axis = 0; axis < d; ++axis) {
        const float delta = query[axis] - center[axis];
        centered_query[axis] = delta;
        center_distance_squared += delta * delta;
    }
    query_pivot_squared[0] = center_distance_squared;
    if (active_count == 1) {
        *sqrt_rho_query =
            std::sqrt(std::max(0.0f, center_distance_squared));
        return std::isfinite(*sqrt_rho_query);
    }

    const size_t n = active_count - 1;
    float z_norm = 0.0f;
    for (size_t row = 0; row < n; ++row) {
        float projection = 0.0f;
#pragma omp simd reduction(+ : projection)
        for (size_t axis = 0; axis < d; ++axis) {
            projection += centered_query[axis] * metadata.U[row * d + axis];
        }
        z_query[row] = projection;
        z_norm += projection * projection;
    }
    *sqrt_rho_query = std::sqrt(std::max(0.0f, center_distance_squared - z_norm));
    return std::isfinite(*sqrt_rho_query);
}

float signatureLowerBoundSquared(const float* z_query,
                                 float sqrt_rho_query,
                                 const PivotMetadata& metadata,
                                 size_t candidate_index,
                                 size_t active_count) {
    const size_t n = active_count - 1;
    float parallel = 0;
    for (size_t i = 0; i < n; ++i) {
        float err = 0;
        const float ze = metadata.signatureValue(active_count, i, candidate_index, &err);
        const float abs_diff = std::abs(z_query[i] - ze);
        const float slack = std::max(0.0f, abs_diff - err);
        parallel += slack * slack;
    }
    float err_tail = 0;
    float sqrt_rho_e = metadata.signatureValue(active_count, n, candidate_index, &err_tail);
    // Move decoded residual toward query residual to avoid over-pruning.
    if (sqrt_rho_e > sqrt_rho_query) sqrt_rho_e = std::max(sqrt_rho_query, sqrt_rho_e - err_tail);
    else sqrt_rho_e = std::min(sqrt_rho_query, sqrt_rho_e + err_tail);
    const float gap = sqrt_rho_query - sqrt_rho_e;
    const float bound = parallel + gap * gap;
    if (!std::isfinite(bound) || bound < -1e-4f) return 0;
    return std::nextafter(std::max(0.0f, bound), 0.0f);
}

bool signaturePrunes(const float* z_query,
                     float sqrt_rho_query,
                     const PivotMetadata& metadata,
                     size_t candidate_index,
                     size_t active_count,
                     float radius) {
    if (metadata.signature_precision == SignaturePrecision::FLOAT32 &&
        !metadata.signature_data.empty()) {
        return signaturePrunes(z_query, sqrt_rho_query,
                               metadata.signature_data.data(), candidate_index,
                               metadata.signature_candidate_count, active_count, radius);
    }
    // Conservative quantized LB: Σ max(0, |q-ẽ|-δ)^2 (+ residual with slack).
    return signatureLowerBoundSquared(z_query, sqrt_rho_query, metadata, candidate_index,
                                       active_count) > radius;
}

uint8_t signaturePruneMask8(const float* z_query,
                            float sqrt_rho_query,
                            const PivotMetadata& metadata,
                            size_t base_index,
                            size_t active_count,
                            float radius) {
    if (metadata.signature_precision == SignaturePrecision::FLOAT32 &&
        !metadata.signature_data.empty()) {
        return signaturePruneMask8(z_query, sqrt_rho_query,
                                   metadata.signature_data.data(), base_index,
                                   metadata.signature_candidate_count, active_count, radius);
    }
    uint8_t mask = 0;
    for (int lane = 0; lane < 8; ++lane) {
        if (signaturePrunes(z_query, sqrt_rho_query, metadata, base_index + lane, active_count,
                            radius)) {
            mask |= static_cast<uint8_t>(1u << lane);
        }
    }
    return mask;
}

double maxSinglePivotLowerBoundSquared(const double* query_pivot_squared,
                                       const double* candidate_pivot_squared,
                                       size_t count) {
    double bound = 0;
    for (size_t i = 0; i < count; ++i) {
        const double delta = std::sqrt(std::max(0.0, query_pivot_squared[i])) -
                             std::sqrt(std::max(0.0, candidate_pivot_squared[i]));
        bound = std::max(bound, delta * delta);
    }
    return std::nextafter(bound, 0.0);
}

double affineProjectionLowerBoundSquared(const PivotMetadata& metadata,
                                         const double* query_pivot_squared,
                                         const double* candidate_pivot_squared,
                                         size_t active_count,
                                         bool* valid) {
    *valid = metadata.usable_projection(active_count);
    if (!*valid) return 0;

    // ||q-x||^2 = ||P_V(q-x)||^2 + ||q_perp - x_perp||^2
    //           >= ||P_V(q-x)||^2 + (||q_perp|| - ||x_perp||)^2
    // Heights: h = ||.-v0||^2 - ||P_V(.-v0)||^2.  P=1 has no parallel span.
    const size_t n = active_count - 1;
    if (n > 0 && metadata.gram_pinv.size() != n * n) {
        *valid = false;
        return 0;
    }

    const double* pinv = n == 0 ? nullptr : metadata.gram_pinv.data();
    std::vector<double> b_query(n);
    std::vector<double> b_candidate(n);
    std::vector<double> delta_b(n);
    for (size_t i = 0; i < n; ++i) {
        const double vi2 = metadata.pairwise_squared[i + 1];
        b_query[i] = 0.5 * (query_pivot_squared[0] + vi2 - query_pivot_squared[i + 1]);
        b_candidate[i] = 0.5 * (candidate_pivot_squared[0] + vi2 - candidate_pivot_squared[i + 1]);
        delta_b[i] = b_query[i] - b_candidate[i];
    }

    auto quadratic = [&](const std::vector<double>& lhs, const std::vector<double>& rhs,
                         double* absolute_sum) {
        double value = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const double term = lhs[i] * pinv[i * n + j] * rhs[j];
                value += term;
                *absolute_sum += std::abs(term);
            }
        }
        return value;
    };

    double absolute_sum = 0;
    const double parallel = n == 0 ? 0.0 : quadratic(delta_b, delta_b, &absolute_sum);
    double query_abs = 0;
    double candidate_abs = 0;
    const double query_parallel = n == 0 ? 0.0 : quadratic(b_query, b_query, &query_abs);
    const double candidate_parallel =
        n == 0 ? 0.0 : quadratic(b_candidate, b_candidate, &candidate_abs);
    absolute_sum += query_abs + candidate_abs;

    const double query_height = std::max(0.0, query_pivot_squared[0] - query_parallel);
    const double candidate_height =
        std::max(0.0, candidate_pivot_squared[0] - candidate_parallel);
    const double height_gap = std::sqrt(query_height) - std::sqrt(candidate_height);
    const double perpendicular = height_gap * height_gap;
    const double bound = parallel + perpendicular;

    if (!std::isfinite(bound) || !std::isfinite(query_height) || !std::isfinite(candidate_height) ||
        bound < -1e-8) {
        *valid = false;
        return 0;
    }
    // Rank truncation only removes projected components. Subtract a
    // condition-scaled forward-error envelope before downward rounding so the
    // floating-point value remains conservative as well as the exact formula.
    const double condition = n == 0 ? 1.0 : metadata.condition;
    const double safe_condition =
        std::isfinite(condition) && condition > 0.0 ? condition : 1.0;
    const double error = 128 * std::numeric_limits<double>::epsilon() *
                         std::max(1.0, safe_condition) *
                         std::max(1.0, absolute_sum + query_height + candidate_height + perpendicular);
    return std::nextafter(std::max(0.0, bound - error), 0.0);
}

void PivotMetadata::save(std::ostream& out) const {
    if (signature_candidate_count != 0 &&
        signature_data.size() != signature_candidate_count * count) {
        throw std::runtime_error(
            "Cannot persist compressed-only signatures; save canonical float32 metadata");
    }
    if (candidate_centroid_squared.size() != 0 &&
        candidate_centroid_squared.size() != signature_candidate_count) {
        throw std::runtime_error("Candidate centroid distance table has invalid size");
    }
    writeScalar<uint32_t>(out, PIVOT_BLOCK_MAGIC);
    writeScalar<uint32_t>(out, MULTIPIVOT_FORMAT_VERSION);
    writeScalar<uint64_t>(out, requested_count);
    writeScalar<uint64_t>(out, count);
    writeScalar<uint64_t>(out, d);
    writeScalar<uint64_t>(out, rank);
    writeScalar<double>(out, condition);
    writeScalar<uint64_t>(out, fallback_count);
    writeScalar<uint8_t>(out, external_projection_geometry ? 1 : 0);
    // A global-projection list owns only its candidate signature; its center
    // and basis live in Index::global_pivots. Per-list metadata retains its
    // center so it can materialize query projections independently.
    if ((!external_projection_geometry && d != 0 && codes.size() < d) ||
        (!codes.empty() && codes.size() < d)) {
        throw std::runtime_error("Pivot center has invalid size");
    }
    std::vector<float> center;
    if (!codes.empty()) center.assign(codes.begin(), codes.begin() + d);
    writeVector(out, center);
    writeVector(out, U);
    writeScalar<uint64_t>(out, signature_candidate_count);
    writeVector(out, signature_data);
    writeVector(out, candidate_centroid_squared);
}

void PivotMetadata::load(std::istream& in, size_t expected_d, size_t expected_candidates) {
    if (readScalar<uint32_t>(in) != PIVOT_BLOCK_MAGIC) {
        throw std::runtime_error("Unsupported pivot metadata format");
    }
    const uint32_t version = readScalar<uint32_t>(in);
    if (version != 10 && version != MULTIPIVOT_FORMAT_VERSION) {
        throw std::runtime_error("Unsupported pivot metadata format");
    }
    requested_count = readScalar<uint64_t>(in);
    count = readScalar<uint64_t>(in);
    d = readScalar<uint64_t>(in);
    rank = readScalar<uint64_t>(in);
    condition = readScalar<double>(in);
    fallback_count = readScalar<uint64_t>(in);
    external_projection_geometry =
        version >= 11 && readScalar<uint8_t>(in) != 0;
    if (count > MULTIPIVOT_MAX_PIVOTS ||
        (count != 0 && d != expected_d) ||
        (count == 0 && d != 0 && d != expected_d) ||
        requested_count > MULTIPIVOT_MAX_PIVOTS) {
        throw std::runtime_error("Invalid pivot metadata dimensions");
    }
    source_ids.clear();
    pairwise_squared.clear();
    candidate_pivot_squared.clear();
    gram_pinv.clear();
    gram_eigenvalues.clear();
    transform_T.clear();
    const size_t geometry_n = count == 0 ? 0 : count - 1;
    readVector(in, codes);
    readVector(in, U);
    const size_t expected_transform = geometry_n * geometry_n;
    const size_t expected_U = geometry_n * d;
    if ((codes.size() != 0 && codes.size() != d) ||
        (!transform_T.empty() && transform_T.size() != expected_transform) ||
        (!U.empty() && U.size() != expected_U) ||
        (!external_projection_geometry && count > 1 && transform_T.empty() && U.empty())) {
        throw std::runtime_error("Invalid pivot projection geometry");
    }
    signature_candidate_count = readScalar<uint64_t>(in);
    readVector(in, signature_data, expected_candidates * count);
    // P=0 indexes intentionally carry no pivot metadata, including no
    // centroid-distance table.  expected_candidates still describes the IVF
    // list size, so requiring that many entries would make a clean P=0 cache
    // impossible to reload.
    readVector(in, candidate_centroid_squared,
               count == 0 ? 0 : expected_candidates);
    if (count != 0 && expected_candidates != 0 &&
        signature_candidate_count != expected_candidates) {
        throw std::runtime_error("Invalid pivot signature table");
    }
}

std::string joinPivotIds(const PivotMetadata& metadata, size_t active_count) {
    std::ostringstream out;
    for (size_t i = 0; i < std::min(active_count, metadata.source_ids.size()); ++i) {
        if (i) out << ';';
        out << metadata.source_ids[i];
    }
    return out.str();
}

std::string joinPairwiseDistances(const PivotMetadata& metadata, size_t active_count) {
    std::vector<double> values;
    const size_t count = std::min(active_count, metadata.count);
    // Compact v10 indexes intentionally do not persist the diagnostic
    // pairwise-distance matrix. An empty manifest field is the correct loaded
    // representation; never index a matrix that is absent.
    if (metadata.pairwise_squared.size() < metadata.count * metadata.count) {
        return "";
    }
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            values.push_back(std::sqrt(std::max(0.0, metadata.pairwise_squared[i * metadata.count + j])));
        }
    }
    return joinDoubles(values);
}

std::string joinEigenvalues(const PivotMetadata& metadata, size_t active_count) {
    if (active_count == metadata.count) return joinDoubles(metadata.gram_eigenvalues);
    return "";
}

}  // namespace tribase
