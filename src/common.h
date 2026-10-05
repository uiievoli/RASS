#pragma once

#ifndef NDEBUG
#define DEBUG
// #define CORRECTNESS_CHECK
#endif

#ifdef TRIBASE_ENABLE_STATS
#define IF_STATS if (stats)
#else
#define IF_STATS if (false)
#endif

#include <inttypes.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace tribase {
inline constexpr uint32_t INDEX_FORMAT_VERSION = 10;

enum MetricType { METRIC_IP = 0,
                  METRIC_L2 };

enum OptLevel {
    OPT_NONE = 0b000,
    OPT_TRIANGLE = 0b001,
    OPT_SUBNN_L2 = 0b010,
    OPT_SUBNN_IP = 0b100,
    OPT_TRI_SUBNN_L2 = 0b011,
    OPT_TRI_SUBNN_IP = 0b101,
    OPT_SUBNN_ONLY = 0b110,
    OPT_ALL = 0b111
};

// Multi-pivot filtering is deliberately independent of OptLevel's legacy
// three-bit mask so existing optimization combinations retain their meaning.
enum class MultiPivotMode : uint32_t {
    NONE = 0,
    MAX = 1,
    PROJECTION = 2,
};

enum class MultiPivotScope : uint32_t {
    GLOBAL = 0,
    PER_LIST = 1,
};

// Storage precision for per-candidate signature tables Φ(e)=(z, √ρ).
enum class SignaturePrecision : uint32_t {
    FLOAT32 = 0,
    FLOAT16 = 1,
    FLOAT8 = 2,  // per-dimension int8 + scale (8-bit quantized)
};

inline const char* multiPivotModeName(MultiPivotMode mode) {
    switch (mode) {
        case MultiPivotMode::NONE: return "none";
        case MultiPivotMode::MAX: return "max";
        case MultiPivotMode::PROJECTION: return "projection";
    }
    return "unknown";
}

inline const char* multiPivotScopeName(MultiPivotScope scope) {
    return scope == MultiPivotScope::GLOBAL ? "global" : "per_list";
}

inline const char* signaturePrecisionName(SignaturePrecision precision) {
    switch (precision) {
        case SignaturePrecision::FLOAT32: return "float32";
        case SignaturePrecision::FLOAT16: return "float16";
        case SignaturePrecision::FLOAT8: return "float8";
    }
    return "unknown";
}

inline bool validSignaturePrecisionName(const std::string& name) {
    return name == "float32" || name == "fp32" || name == "f32" || name == "float16" ||
           name == "fp16" || name == "f16" || name == "float8" || name == "fp8" || name == "f8" ||
           name == "int8";
}

inline SignaturePrecision parseSignaturePrecision(const std::string& name) {
    if (name == "float32" || name == "fp32" || name == "f32") return SignaturePrecision::FLOAT32;
    if (name == "float16" || name == "fp16" || name == "f16") return SignaturePrecision::FLOAT16;
    if (name == "float8" || name == "fp8" || name == "f8" || name == "int8") {
        return SignaturePrecision::FLOAT8;
    }
    throw std::invalid_argument("Invalid signature precision: " + name);
}

enum EdgeDevice {
    EDGEDEVIVE_ENABLED,
    EDGEDEVIVE_DISABLED
};

inline bool isLegalOptLevel(int opt) {
    switch (opt) {
        case OPT_NONE:
        case OPT_TRIANGLE:
        case OPT_SUBNN_L2:
        case OPT_SUBNN_IP:
        case OPT_TRI_SUBNN_L2:
        case OPT_TRI_SUBNN_IP:
        case OPT_SUBNN_ONLY:
        case OPT_ALL:
            return true;
        default:
            return false;
    }
}

inline OptLevel str2OptLevel(const std::string& str) {
    try {
        int int_opt = std::stoi(str);
        if (!isLegalOptLevel(int_opt)) {
            throw std::invalid_argument("Invalid optimization level");
        }
        return static_cast<OptLevel>(int_opt);
    } catch (const std::invalid_argument& e) {
        // pass
    }
    if (str == "OPT_NONE") {
        return OptLevel::OPT_NONE;
    } else if (str == "OPT_TRIANGLE") {
        return OptLevel::OPT_TRIANGLE;
    } else if (str == "OPT_SUBNN_L2") {
        return OptLevel::OPT_SUBNN_L2;
    } else if (str == "OPT_SUBNN_IP") {
        return OptLevel::OPT_SUBNN_IP;
    } else if (str == "OPT_TRI_SUBNN_L2") {
        return OptLevel::OPT_TRI_SUBNN_L2;
    } else if (str == "OPT_TRI_SUBNN_IP") {
        return OptLevel::OPT_TRI_SUBNN_IP;
    } else if (str == "OPT_SUBNN_ONLY") {
        return OptLevel::OPT_SUBNN_ONLY;
    } else if (str == "OPT_ALL") {
        return OptLevel::OPT_ALL;
    } else {
        throw std::runtime_error("Invalid optimization level");
    }
}

using idx_t = int64_t;
using result_t = std::pair<float, idx_t>;

inline thread_local size_t trace_query_id = 0;
inline thread_local size_t trace_probe_rank = 0;
inline thread_local idx_t trace_list_id = -1;

inline void setTraceContext(size_t query_id, size_t probe_rank, idx_t list_id) {
    trace_query_id = query_id;
    trace_probe_rank = probe_rank;
    trace_list_id = list_id;
}

inline bool traceEnabled() {
    const char* value = std::getenv("TRIBASE_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline uint64_t traceMaxLines() {
    const char* value = std::getenv("TRIBASE_TRACE_MAX_LINES");
    if (value == nullptr || value[0] == '\0') return 2000;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    return end == value ? 2000 : static_cast<uint64_t>(parsed);
}

inline bool traceReserveLine() {
    if (!traceEnabled()) return false;
    static std::atomic<uint64_t> lines{0};
    uint64_t old = lines.fetch_add(1, std::memory_order_relaxed);
    return old < traceMaxLines();
}

inline void traceLog(const char* fmt, ...) {
    if (!traceReserveLine()) return;
    std::fprintf(stderr, "TRIBASE_TRACE,");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

}  // namespace tribase
