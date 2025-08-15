#ifndef {{ project_name }}_STREAM_H_
#define {{ project_name }}_STREAM_H_

#include <ap_int.h>         // ap_uint<32>
#include <ap_axi_sdata.h>   // ap_axiu + AXIS_ENABLE_* flags
#include <hls_stream.h>

#include "{{ project_name }}.h"   // generated network core
#include "defines.h"

// 16-bit AXI4-Stream word with KEEP + LAST (no STRB/USER/ID/DEST)
typedef ap_axiu<16, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST)> axis_word_t;

// Προαιρετικό: AXI side-channel για μετρήσεις (χρήσιμο στην cosim)
typedef ap_axiu<32, 0, 0, 0> perf_word_t;

/**
 * Top-level pure AXI4-Stream wrapper με ενσωματωμένο cycle counter.
 *
 * Χρήση:
 *  - Για hardware με GPIO: ΔΕΝ ορίζεις το macro USE_AXIS_PERF
 *    και παίρνεις 2 scalars (cycles_core_out, cycles_e2e_out).
 *  - Για C/RTL cosim: ΟΡΙΖΕΙΣ -DUSE_AXIS_PERF και παίρνεις
 *    δεύτερη έξοδο AXI (m_axis_perf) με 2 beats (core, e2e).
 */
void {{ project_name }}_stream(
    hls::stream<axis_word_t> &s_axis,
    hls::stream<axis_word_t> &m_axis,
#ifdef USE_AXIS_PERF
    hls::stream<perf_word_t> &m_axis_perf   // 2 beats: core then e2e (TLAST στο 2ο)
#else
    ap_uint<32> &cycles_core_out,           // core-only cycles (GPIO-friendly)
    ap_uint<32> &cycles_e2e_out             // end-to-end cycles (GPIO-friendly)
#endif
);

#endif // {{ project_name }}_STREAM_H_
