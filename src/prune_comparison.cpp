#include "Index.h"
#include "heap.hpp"
#include "utils.h"
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace tribase;

// Offline, exhaustive bound-quality experiment. Uses the top-k distance in
// the visited candidate set, not the global ground truth, as a common tau.
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: prune_comparison TRIANGLE_INDEX QUERY_FVECS GT_BIN OUTPUT_DIR\n";
        return 2;
    }
    const std::filesystem::path outdir(argv[4]);
    std::filesystem::create_directories(outdir);
    Index index;
    index.load_index(argv[1]);
    if (index.metric != MetricType::METRIC_L2 || index.d < 64 || index.nlist < 64)
        throw std::runtime_error("Requires L2 and at least 64 dimensions/lists");
    auto [queries, available, d] = loadXvecs(argv[2]);
    if (static_cast<size_t>(d) != index.d) throw std::runtime_error("Dimension mismatch");
    const size_t nq = std::min<size_t>(1000, available), k = 10;
    constexpr std::array<size_t, 3> probes{16, 32, 64};
    constexpr std::array<size_t, 6> dimensions{4, 8, 16, 24, 32, 64};
    std::vector<std::vector<size_t>> listids(nq);
    std::vector<float> qc2(nq * 64);
    std::vector<float> taus(nq * 3), base_dist(nq * 3 * k);
    std::vector<idx_t> base_ids(nq * 3 * k), gt(nq * k);
    std::vector<float> gt_dist(nq * k);
    loadResults(argv[3], gt.data(), gt_dist.data(), nq, k);
#pragma omp parallel for
    for (size_t qi = 0; qi < nq; ++qi) {
        const float* q = queries.get() + qi * index.d;
        std::vector<std::pair<float, size_t>> coarse(index.nlist);
        for (size_t l = 0; l < index.nlist; ++l)
            coarse[l] = {calculatedEuclideanDistance(q, index.centroid_codes.get() + l * index.d, index.d), l};
        std::partial_sort(coarse.begin(), coarse.begin() + 64, coarse.end());
        listids[qi].resize(64);
        std::vector<float> hd(k, std::numeric_limits<float>::max());
        std::vector<idx_t> hi(k, -1);
        for (size_t rank = 0; rank < 64; ++rank) {
            const size_t l = coarse[rank].second;
            listids[qi][rank] = l;
            qc2[qi * 64 + rank] = coarse[rank].first;
            const auto& list = index.lists[l];
            for (size_t x = 0; x < list.list_size; ++x) {
                const float dist = calculatedEuclideanDistance(q, list.candidate_codes.get() + x * index.d, index.d);
                if (dist < hd[0]) heap_replace_top<MetricType::METRIC_L2>(k, hd.data(), hi.data(), dist, list.candidate_id[x]);
            }
            for (size_t p = 0; p < probes.size(); ++p) if (rank + 1 == probes[p]) {
                taus[p * nq + qi] = hd[0];
                auto td = hd; auto ti = hi;
                heap_sort<MetricType::METRIC_L2>(k, td.data(), ti.data());
                std::copy(td.begin(), td.end(), base_dist.begin() + (p * nq + qi) * k);
                std::copy(ti.begin(), ti.end(), base_ids.begin() + (p * nq + qi) * k);
            }
        }
    }
    std::ofstream out(outdir / "oracle.csv");
    out << "scope,nprobe,m,pivot_count,recall,candidates,triangle_pruned,ppd_prefix_pruned,proj_pruned,triangle_ppd_pruned,triangle_proj_pruned,unsafe_ppd,unsafe_proj\n";
    for (auto scope : {MultiPivotScope::GLOBAL, MultiPivotScope::PER_LIST}) {
        const std::string name = scope == MultiPivotScope::GLOBAL ? "global" : "per_list";
        const auto rich = outdir / (name + "_P65.index");
        if (std::filesystem::exists(rich)) index.load_index(rich.string());
        else {
            std::cout << "Building shared " << name << " PCA P65" << std::endl;
            index.configure_multipivot(scope, "pca", 65, 0);
            index.rebuild_multipivot_metadata();
            index.save_index(rich.string());
        }
        for (size_t p = 0; p < probes.size(); ++p) {
            std::array<uint64_t, 6> ppd{}, proj{}, tri_ppd{}, tri_proj{}, unsafe_ppd{}, unsafe_proj{};
            uint64_t candidates = 0, triangles = 0;
#pragma omp parallel
            {
                std::array<uint64_t, 6> a{}, b{}, c{}, e{}, up{}, ub{};
                uint64_t nc = 0, nt = 0;
#pragma omp for schedule(dynamic, 1)
                for (size_t qi = 0; qi < nq; ++qi) {
                    const float* q = queries.get() + qi * index.d;
                    const double tau2 = taus[p * nq + qi];
                    const double tau = std::sqrt(tau2);
                    for (size_t rank = 0; rank < probes[p]; ++rank) {
                        const auto& list = index.lists[listids[qi][rank]];
                        const auto& meta = list.pivots;
                        const auto& geom = scope == MultiPivotScope::GLOBAL ? index.global_pivots : meta;
                        std::array<double, 64> qz{}, hq{};
                        double qr2 = 0;
                        for (size_t ax = 0; ax < index.d; ++ax) {
                            double delta = static_cast<double>(q[ax]) - geom.codes[ax];
                            qr2 += delta * delta;
                        }
                        double residual = qr2;
                        for (size_t dim = 0; dim < 64; ++dim) {
                            for (size_t ax = 0; ax < index.d; ++ax)
                                qz[dim] += (static_cast<double>(q[ax]) - geom.codes[ax]) * geom.U[dim * index.d + ax];
                            residual -= qz[dim] * qz[dim];
                            hq[dim] = std::sqrt(std::max(0.0, residual));
                        }
                        for (size_t x = 0; x < list.list_size; ++x) {
                            ++nc;
                            const double tr = std::abs(std::sqrt(qc2[qi * 64 + rank]) - list.sqrt_candidate2centroid[x]);
                            const bool tri = tr > tau + 1e-5 * std::max(1.0, tau);
                            nt += tri;
                            double xr2 = meta.candidate_centroid_squared.empty()
                                ? meta.candidate_pivot_squared[x * meta.count] : meta.candidate_centroid_squared[x];
                            double score = 0;
                            size_t checkpoint = 0;
                            const double exact = calculatedEuclideanDistance(q, list.candidate_codes.get() + x * index.d, index.d);
                            const double threshold = tau2 + 1e-5 * std::max(1.0, tau2);
                            for (size_t dim = 0; dim < 64; ++dim) {
                                const double xz = meta.signatureValue(65, dim, x);
                                score += (qz[dim] - xz) * (qz[dim] - xz);
                                xr2 -= xz * xz;
                                if (dim + 1 == dimensions[checkpoint]) {
                                    const double delta = hq[dim] - std::sqrt(std::max(0.0, xr2));
                                    const double lb = score + delta * delta;
                                    const bool pa = score > threshold, pb = lb > threshold;
                                    a[checkpoint] += pa; b[checkpoint] += pb;
                                    c[checkpoint] += tri || pa; e[checkpoint] += tri || pb;
                                    up[checkpoint] += pa && exact <= threshold;
                                    ub[checkpoint] += pb && exact <= threshold;
                                    if (++checkpoint == dimensions.size()) break;
                                }
                            }
                        }
                    }
                }
#pragma omp critical
                {
                    candidates += nc; triangles += nt;
                    for (size_t j = 0; j < 6; ++j) {
                        ppd[j] += a[j]; proj[j] += b[j]; tri_ppd[j] += c[j]; tri_proj[j] += e[j];
                        unsafe_ppd[j] += up[j]; unsafe_proj[j] += ub[j];
                    }
                }
            }
            const float recall = calculate_recall(base_ids.data() + p * nq * k, base_dist.data() + p * nq * k, gt.data(), gt_dist.data(), nq, k, MetricType::METRIC_L2);
            for (size_t j = 0; j < 6; ++j)
                out << name << ',' << probes[p] << ',' << dimensions[j] << ',' << dimensions[j] + 1 << ',' << recall << ',' << candidates << ',' << triangles << ',' << ppd[j] << ',' << proj[j] << ',' << tri_ppd[j] << ',' << tri_proj[j] << ',' << unsafe_ppd[j] << ',' << unsafe_proj[j] << '\n';
            out.flush();
            std::cout << name << " nprobe=" << probes[p] << " candidates=" << candidates << std::endl;
        }
    }
}
