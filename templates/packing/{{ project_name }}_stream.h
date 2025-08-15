#ifndef {{ project_name }}_STREAM_H_
#define {{ project_name }}_STREAM_H_

#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "{{ project_name }}.h"
#include "defines.h"

// 64-bit data + KEEP + LAST
typedef ap_axiu<64, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST)> axis64_t;
// 16-bit data + KEEP + LAST
typedef ap_axiu<16, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST)> axis16_t;

// Optional perf side-channel (32-bit words)
typedef ap_axiu<32, 0, 0, 0> perf_word_t;

/**
 * Top-level AXI4-Stream wrapper (64b IN, 16b OUT) with integrated cycle counter.
 *
 * - Default (hardware): expose two scalar outputs (GPIO-friendly).
 * - With -DUSE_AXIS_PERF: expose AXIS perf stream (2 beats: core, e2e).
 */
void {{ project_name }}_stream(
    hls::stream<axis64_t> &s_axis,
    hls::stream<axis16_t> &m_axis,
#ifdef USE_AXIS_PERF
    hls::stream<perf_word_t> &m_axis_perf
#else
    ap_uint<32> &cycles_core_out,
    ap_uint<32> &cycles_e2e_out
#endif
);

#endif // {{ project_name }}_STREAM_H_
