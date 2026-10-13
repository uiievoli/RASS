#include "Index.h"
#include "Ppd.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace tribase;

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: import_ppd_basis SOURCE_INDEX PPD_SIDECAR OUTPUT_INDEX P\n";
            return 2;
        }
        const size_t p = std::stoull(argv[4]);
        Index index;
        index.load_index(argv[1]);
        if (p < 2 || p > MULTIPIVOT_MAX_PIVOTS || p > index.d + 1)
            throw std::invalid_argument("P must be in [2, min(512, D+1)]");
        PpdIndex ppd;
        ppd.load(argv[2], index);
        size_t total = 0;
        for (size_t lid = 0; lid < index.nlist; ++lid) total += index.lists[lid].list_size;
        if (ppd.training_samples != total)
            throw std::invalid_argument("Controlled comparison requires full-training PPD");
        const size_t m = p - 1;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            basis(ppd.basis.data(), m, index.d);
        const Eigen::MatrixXf gram = basis * basis.transpose();
        const float error = (gram - Eigen::MatrixXf::Identity(m, m)).cwiseAbs().maxCoeff();
        if (!basis.allFinite() || !std::isfinite(error) || error > 1e-4f)
            throw std::runtime_error("PPD prefix is not approximately orthonormal");
        index.configure_multipivot(MultiPivotScope::GLOBAL, "pca", p, 0);
        index.pca_cov_eigenvalues.clear();
        index.irls_objective_history.clear();
        index.microblock_enabled = false;
        PivotMetadata metadata;
        metadata.d = index.d;
        metadata.count = metadata.requested_count = p;
        metadata.rank = m;
        metadata.condition = 1.0;
        metadata.codes = ppd.mean;
        metadata.U.assign(ppd.basis.begin(), ppd.basis.begin() + m * index.d);
        index.global_pivots = std::move(metadata);
        // Reuse the exact PPD candidate coordinates. No retraining, rotation or
        // candidate reordering; radius/residual tables remain native float32.
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t lid = 0; lid < index.nlist; ++lid) {
            auto& list = index.lists[lid];
            PivotMetadata candidate;
            candidate.d = index.d;
            candidate.count = candidate.requested_count = p;
            candidate.rank = m;
            candidate.condition = 1.0;
            candidate.external_projection_geometry = true;
            candidate.signature_candidate_count = list.list_size;
            candidate.signature_data.resize(list.list_size * p);
            candidate.candidate_centroid_squared.resize(list.list_size);
            for (size_t i = 0; i < list.list_size; ++i) {
                const float* x = list.get_candidate_codes(i);
                double norm2 = 0;
                for (size_t axis = 0; axis < index.d; ++axis) {
                    const float delta = x[axis] - ppd.mean[axis];
                    norm2 += double(delta) * delta;
                }
                candidate.candidate_centroid_squared[i] = static_cast<float>(norm2);
                float energy = 0;
                for (size_t dim = 0; dim < m; ++dim) {
                    const float z = ppd.transformed_lists[lid][i * index.d + dim];
                    candidate.signature_data[dim * list.list_size + i] = z;
                    energy += z * z;
                }
                candidate.signature_data[m * list.list_size + i] =
                    static_cast<float>(std::sqrt(std::max(0.0, norm2 - energy)));
            }
            list.pivots = std::move(candidate);
            list.microblocks.clear();
        }
        index.apply_signature_precision();
        const std::string partial = std::string(argv[3]) + ".partial";
        index.save_index(partial);
        std::filesystem::rename(partial, argv[3]);
        std::cout << "Imported exact PPD mean/basis/coordinates; P=" << p
                  << " candidates=" << total << " gram_max_error=" << error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
