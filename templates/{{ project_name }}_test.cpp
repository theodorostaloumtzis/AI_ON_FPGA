#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <cstdlib>

#include "firmware/{{ project_name }}_stream.h"   // axis_word_t + top prototype
#include "firmware/nnet_utils/nnet_helpers.h"
#include "ap_fixed.h"
#include "ap_int.h"            // ap_uint<32>
#include "hls_stream.h"
#include "firmware/defines.h"

#define CHECKPOINT 5000
#define N_INPUT 784
#define N_OUTPUT 10

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void*> *trace_outputs = nullptr;
size_t trace_type_size = sizeof(double);
}

// ---------------------------------------------------------------------------
// Helpers to cast between float and fixed-point axis words
// ---------------------------------------------------------------------------
static inline axis_word_t float_to_axis(float x, bool last = false)
{
    axis_word_t w;
    ap_fixed<16,6> fx = x;          // float → Q6.10 cast
    w.data = fx.range(15,0);
    w.keep = 0x3;                   // both bytes valid
    w.last = last;
    return w;
}

static inline float axis_to_float(axis_word_t w)
{
    ap_fixed<16,6> fx;  fx.range(15,0) = w.data;
    return (float)fx;
}

// Simple line-splitter
template<typename T>
static std::vector<T> split_line(const std::string &s)
{
    std::vector<T> elems; std::stringstream ss(s); T v;
    while (ss >> v) elems.push_back(v);
    return elems;
}

int main()
{
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string log_file = "tb_data/rtl_cosim_results.log";
#else
    std::string log_file = "tb_data/csim_results.log";
#endif
    std::ofstream fout(log_file);

    std::string iline, pline; unsigned sample = 0;

    while (std::getline(fin, iline) && std::getline(fpr, pline)) {
        if (sample % CHECKPOINT == 0)
            std::cout << "Processing input " << sample << '\n';

        auto img    = split_line<float>(iline);
        auto golden = split_line<float>(pline);

        hls::stream<axis_word_t> s_axis("s_axis");
        hls::stream<axis_word_t> m_axis("m_axis");

        const unsigned NPIX = N_INPUT;
        for (unsigned i = 0; i < NPIX; ++i)
            s_axis << float_to_axis(img[i], i == NPIX - 1);

        // --- call wrapper (conditional perf side-channel vs scalars) --------
#ifdef USE_AXIS_PERF
        hls::stream<perf_word_t> m_axis_perf("m_axis_perf");
        {{ project_name }}_stream(s_axis, m_axis, m_axis_perf);

        // Drain 2 beats (core, e2e) χωρίς να αλλάξουμε τα logs
        perf_word_t p0 = m_axis_perf.read();
        perf_word_t p1 = m_axis_perf.read();
        (void)p0; (void)p1;
#else
        ap_uint<32> cycles_core = 0;
        ap_uint<32> cycles_e2e  = 0;
        {{ project_name }}_stream(s_axis, m_axis, cycles_core, cycles_e2e);
        // (Δεν τα τυπώνουμε για να μείνει ίδιο το log format)
#endif

        float preds[N_OUTPUT];
        for (unsigned i = 0; i < N_OUTPUT; ++i)
            preds[i] = axis_to_float(m_axis.read());

        if (sample % CHECKPOINT == 0) {
            std::cout << "Golden:   "; for (float v: golden) std::cout << v << ' '; std::cout << "\nPredicted:";
            for (float v: preds)  std::cout << v << ' '; std::cout << "\n\n";
        }
        for (unsigned i = 0; i < N_OUTPUT; ++i)
            fout << preds[i] << (i == N_OUTPUT - 1 ? '\n' : ' ');

        ++sample;
    }

    fout.close();
    std::cout << "INFO: saved results to " << log_file << '\n';
    return 0;
}
