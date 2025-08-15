// testbench_64in_16out.cpp
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cstring>

#include "firmware/{{ project_name }}_stream.h"   // axis64_t + axis16_t + top prototype
#include "firmware/nnet_utils/nnet_helpers.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "firmware/defines.h"

#define CHECKPOINT 5000
#define N_INPUT    784
#define N_OUTPUT   10
#define PACK_SIZE  4     // 64-bit input = 4×16-bit pixels/beat
#define N_BEATS    (N_INPUT / PACK_SIZE)

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void*> *trace_outputs = nullptr;
size_t trace_type_size = sizeof(double);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
template<typename T>
static std::vector<T> split_line(const std::string &s) {
    std::vector<T> elems; std::stringstream ss(s); T v;
    while (ss >> v) elems.push_back(v);
    return elems;
}

// Pack 4 floats into one 64-bit AXI word (Q6.10 → 16-bit → pack)
static inline axis64_t pack4_to_axis64(const float in[4], bool last=false) {
    axis64_t w;  // fields: data (ap_uint<64>), keep (ap_uint<8>), last (ap_uint<1>)
    ap_uint<64> data = 0;
    for (int i = 0; i < 4; ++i) {
        ap_fixed<16,6> fx = in[i];               // float → Q6.10
        data.range(16*i + 15, 16*i) = fx.range(15,0);
    }
    w.data = data;
    w.keep = 0xFF;                                // 8 bytes valid
    w.last = last;
    return w;
}

// Convert a single 16-bit AXI word back to float (Q6.10 → float)
static inline float axis16_to_float(const axis16_t &w) {
    ap_fixed<16,6> fx; fx.range(15,0) = w.data;
    return (float)fx;
}

int main() {
    // --- Open data files ----------------------------------------------------
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::ifstream fpr("tb_data/tb_output_predictions.dat");
    if (!fin) { std::cerr << "ERROR: cannot open tb_input_features.dat\n"; return 1; }
    if (!fpr) { std::cerr << "ERROR: cannot open tb_output_predictions.dat\n"; return 1; }

#ifdef RTL_SIM
    std::string log_file = "tb_data/rtl_cosim_results.log";
#else
    std::string log_file = "tb_data/csim_results.log";
#endif
    std::ofstream fout(log_file);
    if (!fout) {
        std::cerr << "ERROR: cannot open output log file: " << log_file << "\n";
        return 1;
    }

    std::string iline, pline; unsigned sample = 0;

    while (std::getline(fin, iline) && std::getline(fpr, pline)) {
        if (sample % CHECKPOINT == 0)
            std::cout << "Processing input " << sample << '\n';

        // Parse one sample
        auto img    = split_line<float>(iline);
        auto golden = split_line<float>(pline);

        if (img.size() != N_INPUT) {
            std::cerr << "WARN: input size " << img.size() << " != " << N_INPUT << " (skipping)\n";
            continue;
        }
        if (golden.size() != N_OUTPUT) {
            std::cerr << "WARN: golden size " << golden.size() << " != " << N_OUTPUT << " (skipping)\n";
            continue;
        }

        // DUT interfaces
        hls::stream<axis64_t> s_axis("s_axis");
        hls::stream<axis16_t> m_axis("m_axis");

        // 1) Pack & send 64-bit input (4 pixels per beat)
        for (unsigned b = 0; b < N_BEATS; ++b) {
            float tmp[4] = {
                img[4*b + 0],
                img[4*b + 1],
                img[4*b + 2],
                img[4*b + 3]
            };
            bool last = (b == N_BEATS - 1);
            s_axis << pack4_to_axis64(tmp, last);
        }

        // 2) Run DUT (dual-mode: scalar counters OR perf stream)
#ifdef USE_AXIS_PERF
        hls::stream<perf_word_t> m_axis_perf("m_axis_perf");
        {{ project_name }}_stream(s_axis, m_axis, m_axis_perf);
        // Drain the two perf beats (core cycles, then e2e cycles)
        // We purposefully do NOT print them to keep log format unchanged.
        (void)m_axis_perf.read();
        (void)m_axis_perf.read();
#else
        ap_uint<32> cycles_core = 0;
        ap_uint<32> cycles_e2e  = 0;
        {{ project_name }}_stream(s_axis, m_axis, cycles_core, cycles_e2e);
        // Keep silent to preserve the exact log format HLS4ML expects
        // If you want to see them, uncomment:
        // std::cout << "[cycles] core=" << (unsigned)cycles_core
        //           << " e2e=" << (unsigned)cycles_e2e << "\n";
#endif

        // 3) Read 10 logits (each 16-bit per beat)
        float preds[N_OUTPUT];
        for (unsigned i = 0; i < N_OUTPUT; ++i) {
            axis16_t w = m_axis.read();
            preds[i] = axis16_to_float(w);
        }

        // 4) Print checkpoint diagnostics & write log line
        if (sample % CHECKPOINT == 0) {
            std::cout << "Golden:   ";
            for (float v: golden) std::cout << v << ' ';
            std::cout << "\nPredicted:";
            for (float v: preds)  std::cout << v << ' ';
            std::cout << "\n\n";
        }
        for (unsigned i = 0; i < N_OUTPUT; ++i)
            fout << preds[i] << (i + 1 == N_OUTPUT ? '\n' : ' ');

        ++sample;
    }

    fout.close();
    std::cout << "INFO: saved results to " << log_file << '\n';
    return 0;
}
