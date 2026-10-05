// Stream-sample K vectors from a .bvecs.gz (BIGANN-style) into an uncompressed .bvecs.
// Usage: sample_bvecs_gz <in.bvecs.gz> <out.bvecs> <N_total> <K_sample> [seed]

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

[[noreturn]] static void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

static bool read_exact(gzFile in, void* dst, unsigned n) {
    auto* p = static_cast<unsigned char*>(dst);
    unsigned got = 0;
    while (got < n) {
        const int r = gzread(in, p + got, n - got);
        if (r < 0) die("gzread error");
        if (r == 0) return false;
        got += static_cast<unsigned>(r);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <in.bvecs.gz> <out.bvecs> <N_total> <K_sample> [seed]\n",
                     argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    const uint64_t N = std::strtoull(argv[3], nullptr, 10);
    const uint64_t K = std::strtoull(argv[4], nullptr, 10);
    const uint64_t seed = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 6666ull;
    if (K == 0 || N == 0 || K > N) die("need 0 < K <= N");

    std::fprintf(stderr, "sampling K=%llu from N=%llu seed=%llu\n",
                 (unsigned long long)K, (unsigned long long)N, (unsigned long long)seed);
    std::fprintf(stderr, "expected output ~%.2f GiB\n", (K * 132.0) / (1024.0 * 1024.0 * 1024.0));

    std::vector<uint8_t> selected((N + 7) / 8, 0);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, N - 1);
    uint64_t chosen = 0;
    while (chosen < K) {
        const uint64_t i = dist(rng);
        const uint64_t byte = i >> 3;
        const uint8_t bit = static_cast<uint8_t>(1u << (i & 7ull));
        if ((selected[byte] & bit) == 0) {
            selected[byte] |= bit;
            ++chosen;
            if ((chosen & ((1ull << 20) - 1)) == 0) {
                std::fprintf(stderr, "\rchosen %llu / %llu", (unsigned long long)chosen,
                             (unsigned long long)K);
            }
        }
    }
    std::fprintf(stderr, "\rchosen %llu / %llu\n", (unsigned long long)chosen,
                 (unsigned long long)K);

    gzFile in = gzopen(in_path, "rb");
    if (!in) die("cannot open input gzip");
    // Larger buffer helps sequential throughput.
    gzbuffer(in, 1 << 20);

    FILE* out = std::fopen(out_path, "wb");
    if (!out) die("cannot open output");

    std::vector<uint8_t> payload(128);
    uint64_t written = 0;
    uint64_t scanned = 0;
    const uint64_t report_every = 5'000'000;

    while (scanned < N && written < K) {
        int32_t dim = 0;
        if (!read_exact(in, &dim, sizeof(dim))) {
            std::fprintf(stderr, "\ntruncated at vector %llu (file shorter than N?)\n",
                         (unsigned long long)scanned);
            break;
        }
        if (dim != 128) {
            std::fprintf(stderr, "\nunexpected dim=%d at %llu\n", dim,
                         (unsigned long long)scanned);
            die("expected dim=128");
        }
        if (payload.size() < static_cast<size_t>(dim)) payload.resize(static_cast<size_t>(dim));
        if (!read_exact(in, payload.data(), static_cast<unsigned>(dim))) {
            std::fprintf(stderr, "\ntruncated reading payload at %llu\n",
                         (unsigned long long)scanned);
            break;
        }

        const uint64_t byte = scanned >> 3;
        const uint8_t bit = static_cast<uint8_t>(1u << (scanned & 7ull));
        if (selected[byte] & bit) {
            if (std::fwrite(&dim, 1, 4, out) != 4 ||
                std::fwrite(payload.data(), 1, static_cast<size_t>(dim), out) !=
                    static_cast<size_t>(dim)) {
                die("write failed");
            }
            ++written;
        }
        ++scanned;
        if (scanned % report_every == 0) {
            std::fprintf(stderr, "\rscanned %llu / %llu (%.1f%%), written %llu / %llu",
                         (unsigned long long)scanned, (unsigned long long)N,
                         100.0 * scanned / N, (unsigned long long)written,
                         (unsigned long long)K);
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "\ndone: scanned=%llu written=%llu -> %s\n",
                 (unsigned long long)scanned, (unsigned long long)written, out_path);

    gzclose(in);
    std::fclose(out);

    const std::string meta = std::string(out_path) + ".meta.txt";
    if (FILE* m = std::fopen(meta.c_str(), "w")) {
        std::fprintf(m, "source=%s\n", in_path);
        std::fprintf(m, "N_total=%llu\n", (unsigned long long)N);
        std::fprintf(m, "K_sample=%llu\n", (unsigned long long)K);
        std::fprintf(m, "written=%llu\n", (unsigned long long)written);
        std::fprintf(m, "seed=%llu\n", (unsigned long long)seed);
        std::fprintf(m, "format=bvecs\n");
        std::fprintf(m, "bytes_per_vector=132\n");
        std::fclose(m);
    }
    return written == K ? 0 : 1;
}
