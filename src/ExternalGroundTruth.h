#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "common.h"

namespace tribase {

struct ExternalGroundTruth {
    std::unique_ptr<idx_t[]> ids;
    size_t width = 0;
};

inline ExternalGroundTruth load_external_groundtruth(const std::string& path, size_t requested_nq) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open external ground truth: " + path);
    uint32_t stored_nq = 0, width = 0;
    const bool ivecs = path.ends_with(".ivecs");
    const bool i32bin = path.ends_with(".i32bin");
    if (!ivecs && !i32bin) {
        throw std::invalid_argument("External ground truth must be .ivecs or .i32bin: " + path);
    }
    if (i32bin) {
        input.read(reinterpret_cast<char*>(&stored_nq), sizeof(stored_nq));
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
    } else {
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
        if (!input || width == 0) throw std::runtime_error("Invalid ivecs ground truth: " + path);
        const uint64_t bytes = std::filesystem::file_size(path);
        const uint64_t stride = sizeof(uint32_t) * (static_cast<uint64_t>(width) + 1);
        if (bytes % stride != 0) throw std::runtime_error("Truncated ivecs ground truth: " + path);
        stored_nq = static_cast<uint32_t>(bytes / stride);
        input.seekg(0, std::ios::beg);
    }
    if (!input || width == 0 || requested_nq > stored_nq) {
        throw std::runtime_error("External ground truth shape is smaller than requested queries: " + path);
    }
    auto ids = std::make_unique<idx_t[]>(requested_nq * static_cast<size_t>(width));
    std::vector<int32_t> row(width);
    for (size_t query = 0; query < requested_nq; ++query) {
        if (ivecs) {
            uint32_t row_width = 0;
            input.read(reinterpret_cast<char*>(&row_width), sizeof(row_width));
            if (row_width != width) throw std::runtime_error("Mixed-width ivecs ground truth: " + path);
        }
        input.read(reinterpret_cast<char*>(row.data()),
                   static_cast<std::streamsize>(width * sizeof(int32_t)));
        if (!input) throw std::runtime_error("Truncated external ground truth: " + path);
        for (size_t neighbor = 0; neighbor < width; ++neighbor) {
            ids[query * width + neighbor] = static_cast<idx_t>(row[neighbor]);
        }
    }
    return {std::move(ids), static_cast<size_t>(width)};
}

inline float calculate_id_recall(const idx_t* labels, const idx_t* ground_truth,
                          size_t nq, size_t k, size_t ground_truth_width) {
    if (k > ground_truth_width) {
        throw std::invalid_argument("Search k exceeds external ground-truth width");
    }
    size_t hits = 0;
#pragma omp parallel for reduction(+ : hits)
    for (size_t query = 0; query < nq; ++query) {
        std::unordered_set<idx_t> expected(
            ground_truth + query * ground_truth_width,
            ground_truth + query * ground_truth_width + k);
        for (size_t neighbor = 0; neighbor < k; ++neighbor) {
            hits += expected.contains(labels[query * k + neighbor]);
        }
    }
    return static_cast<float>(hits) / static_cast<float>(nq * k);
}

} // namespace tribase
