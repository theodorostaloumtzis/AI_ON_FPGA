#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <cstdlib>

#include "firmware/{{ project_name }}_stream.h"   // axis64_t + axis16_t + top prototype
#include "firmware/nnet_utils/nnet_helpers.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "firmware/defines.h"

#define CHECKPOINT 5000
#define N_INPUT    784
#define N_OUTPUT   10

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void*> *trace_outputs = nullptr;
size_t trace_type_size = sizeof(double);
}

// ---------------------------------------------------------------------------
// Pack 4 floats into one 64-bit AXI word (Q6.10 → ap_fixed<16,6> → bits)
// ---------------------------------------------------------------------------
static inline axis64_t pack_4floats(const float in[4], bool last=false) {
    axis64_t w;
    for (int i = 0; i < 4; ++i) {
        ap_fixed<16,6> fx = in[i];  // float → Q6.10
        w.data.range(16*i + 15, 16*i) = fx.range(15,0);
    }
    w.keep = 0xFF;  // 8 bytes valid
    w.last = last;
    return w;
}

// ---------------------------------------------------------------------------
// Convert a single 16-bit AXI word back to float (Q6.10 → float)
// ---------------------------------------------------------------------------
static inline float axis16_to_float(const axis16_t &w) {
    ap_fixed<16,6> fx; fx.range(15,0) = w.data;
    return (float)fx;
}

// ---------------------------------------------------------------------------
// Simple whitespace-splitter
// ---------------------------------------------------------------------------
template<typename T>
static std::vector<T> split_line(const std::string &s) {
    std::vector<T> elems; std::stringstream ss(s); T v;
    while (ss >> v) elems.push_back(v);
    return elems;
}

int main() {
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string log_file = "tb_data/rtl_cosim_results.log";
#else
    std::string log_file = "tb_data/csim_results.log";
#endif
    std::ofstream fout(log_file);

    std::string iline, pline;
    unsigned sample = 0;

    while (std::getline(fin, iline) && std::getline(fpr, pline)) {
        if (sample % CHECKPOINT == 0)
            std::cout << "Processing input " << sample << "\n";

        auto img    = split_line<float>(iline);
        auto golden = split_line<float>(pline);

        hls::stream<axis64_t> s_axis("s_axis");
        hls::stream<axis16_t> m_axis("m_axis");

        // 1) Pack & write input in groups of 4 floats
        for (unsigned i = 0; i < N_INPUT; i += 4) {
            float tmp[4];
            for (int j = 0; j < 4; ++j) tmp[j] = img[i + j];
            bool last = (i + 4 >= N_INPUT);
            s_axis << pack_4floats(tmp, last);
        }

        // 2) Run DUT
        {{ project_name }}_stream(s_axis, m_axis);

        // 3) Read & unpack each 16-bit logit
        float preds[N_OUTPUT];
        for (unsigned i = 0; i < N_OUTPUT; ++i) {
            axis16_t w = m_axis.read();
            preds[i]   = axis16_to_float(w);
        }

        // 4) Print/log every CHECKPOINT samples
        if (sample % CHECKPOINT == 0) {
            std::cout << "Golden:   ";
            for (auto v: golden) std::cout << v << ' ';
            std::cout << "\nPredicted:";
            for (auto v: preds)   std::cout << v << ' ';
            std::cout << "\n\n";
        }
        for (unsigned i = 0; i < N_OUTPUT; ++i)
            fout << preds[i] << (i+1==N_OUTPUT?'\n':' ');

        ++sample;
    }

    fout.close();
    std::cout << "INFO: saved results to " << log_file << '\n';
    return 0;
}
