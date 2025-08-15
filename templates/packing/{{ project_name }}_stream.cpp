#include "{{ project_name }}_stream.h"   // axis64_t + axis16_t + top prototype

#define N_INPUT    784   // 28×28
#define N_OUTPUT     10
#define PACK_SIZE     4  // pixels per 64-bit beat
#define N_PKT   (N_INPUT / PACK_SIZE)

#ifndef START_ON_LAST_IN
#define START_ON_LAST_IN  1   // 1: start on input TLAST, 0: start on first beat
#endif

// ───────────────────────── Stages for DATAFLOW ──────────────────────────────
static void ingest_pixels(hls::stream<axis64_t>   &s_axis,
                          hls::stream<input_t>    &img_fifo,
                          hls::stream<ap_uint<1>> &start_evt_s)
{
#pragma HLS INLINE off
READ_PKTS:
    for (unsigned pkt = 0; pkt < N_PKT; ++pkt) {
#pragma HLS PIPELINE II=1
        axis64_t aw = s_axis.read();

        // One start event per frame
        if (START_ON_LAST_IN) {
            if (aw.last) start_evt_s.write(1);
        } else {
            if (pkt == 0) start_evt_s.write(1);
        }

        // Unpack 64b → 4×16b pixels (LSW first)
    UNPACK4:
        for (int j = 0; j < PACK_SIZE; ++j) {
#pragma HLS UNROLL
            input_t pix;
            pix[0].range(15,0) = aw.data.range(16*j + 15, 16*j);
            img_fifo.write(pix);
        }
    }
}

static void run_core(hls::stream<input_t>     &img_fifo,
                     hls::stream<result_t>    &res_fifo,
                     hls::stream<ap_uint<1>>  &done_core_evt_s)
{
#pragma HLS INLINE off
    {{ project_name }}(img_fifo, res_fifo);
    // Signal core-only completion (result is ready)
    done_core_evt_s.write(1);
}

static void egress_logits(hls::stream<result_t>    &res_fifo,
                          hls::stream<axis16_t>    &m_axis,
                          hls::stream<ap_uint<1>>  &done_e2e_evt_s)
{
#pragma HLS INLINE off
    result_t logits = res_fifo.read();

WRITE_LOGITS:
    for (unsigned c = 0; c < N_OUTPUT; ++c) {
#pragma HLS PIPELINE II=1
        axis16_t ow;
        ow.data = logits[c].range(15,0);
        ow.keep = 0x3;                       // 2 bytes valid
        ow.last = (c == N_OUTPUT - 1);
        m_axis.write(ow);

        if (ow.last) {
            // End-to-end completion on last output handshake
            done_e2e_evt_s.write(1);
        }
    }
}

// Single-shot dual counter: counts cycles from start event to core/e2e done
static void cycle_counter_proc(hls::stream<ap_uint<1>> &start_evt_s,
                               hls::stream<ap_uint<1>> &done_core_evt_s,
                               hls::stream<ap_uint<1>> &done_e2e_evt_s,
                               ap_uint<32> &cycles_core_out,
                               ap_uint<32> &cycles_e2e_out)
{
#pragma HLS INLINE off
    ap_uint<1> dummy;
    start_evt_s.read(dummy);   // block for the frame start

    ap_uint<32> cnt_core = 0, cnt_e2e = 0;
    bool run_core = true, run_e2e = true;

COUNT_LOOP:
    while (run_core || run_e2e) {
#pragma HLS PIPELINE II=1
        if (run_core) cnt_core++;
        if (run_e2e)  cnt_e2e++;

        if (run_core && !done_core_evt_s.empty()) {
            (void)done_core_evt_s.read();
            run_core = false;
        }
        if (run_e2e && !done_e2e_evt_s.empty()) {
            (void)done_e2e_evt_s.read();
            run_e2e = false;
        }
    }

    cycles_core_out = cnt_core;
    cycles_e2e_out  = cnt_e2e;
}

// ──────────────────────────── Top-level wrapper ─────────────────────────────
void {{ project_name }}_stream(
    hls::stream<axis64_t> &s_axis,
    hls::stream<axis16_t> &m_axis,
#ifdef USE_AXIS_PERF
    hls::stream<perf_word_t> &m_axis_perf
#else
    ap_uint<32> &cycles_core_out,
    ap_uint<32> &cycles_e2e_out
#endif
)
{
#pragma HLS INTERFACE axis         port=s_axis
#pragma HLS INTERFACE axis         port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return
#ifdef USE_AXIS_PERF
#pragma HLS INTERFACE axis         port=m_axis_perf
#else
#pragma HLS INTERFACE ap_none      port=cycles_core_out
#pragma HLS INTERFACE ap_none      port=cycles_e2e_out
#endif

#pragma HLS DATAFLOW

    // Small decoupling FIFOs (no full-frame buffering required)
    hls::stream<input_t>     img_fifo("img_fifo");
    hls::stream<result_t>    res_fifo("res_fifo");
#pragma HLS STREAM variable=img_fifo depth=64
#pragma HLS STREAM variable=res_fifo depth=16

    // Event FIFOs for the counter
    hls::stream<ap_uint<1>>  start_evt_s("start_evt_s");
    hls::stream<ap_uint<1>>  done_core_evt_s("done_core_evt_s");
    hls::stream<ap_uint<1>>  done_e2e_evt_s("done_e2e_evt_s");
#pragma HLS STREAM variable=start_evt_s     depth=2
#pragma HLS STREAM variable=done_core_evt_s depth=2
#pragma HLS STREAM variable=done_e2e_evt_s  depth=2

    // 1) Ingest 64b → 2) Run core → 3) Egress 16b
    ingest_pixels(s_axis, img_fifo, start_evt_s);
    run_core(img_fifo, res_fifo, done_core_evt_s);
    egress_logits(res_fifo, m_axis, done_e2e_evt_s);

#ifndef USE_AXIS_PERF
    // 4A) Scalar outputs (GPIO-friendly)
    cycle_counter_proc(start_evt_s, done_core_evt_s, done_e2e_evt_s,
                       cycles_core_out, cycles_e2e_out);
#else
    // 4B) AXIS perf side-channel (2 beats: core, e2e)
    ap_uint<32> cyc_core, cyc_e2e;
    cycle_counter_proc(start_evt_s, done_core_evt_s, done_e2e_evt_s,
                       cyc_core, cyc_e2e);

    perf_word_t pw;
    pw.data = cyc_core; pw.keep = ~ap_uint<4>(0); pw.strb = ~ap_uint<4>(0); pw.last = 0;
    m_axis_perf.write(pw);
    pw.data = cyc_e2e;  pw.last = 1;
    m_axis_perf.write(pw);
#endif
}
