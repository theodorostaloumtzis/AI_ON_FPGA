#include "{{ project_name }}_stream.h"   // axis_word_t + top prototype

#define N_INPUT   784   // 28×28
#define N_OUTPUT  10    // 10 classes

// Προαιρετικό: start στο TLAST του input (1) ή στο πρώτο beat (0)
#define START_ON_LAST_IN  1

// ----------------- Μικρά stages για DATAFLOW --------------------------------
static void ingest_pixels(hls::stream<axis_word_t> &s_axis,
                          hls::stream<input_t> &img_fifo,
                          hls::stream<ap_uint<1>> &start_evt_s)
{
#pragma HLS INLINE off
READ_PIXELS:
    for (unsigned i = 0; i < N_INPUT; ++i) {
#pragma HLS PIPELINE II=1
        axis_word_t w = s_axis.read();         // μπλοκάρει σε TVALID∧TREADY

        // start event (μία φορά ανά frame)
        if (START_ON_LAST_IN) {
            if (w.last) start_evt_s.write(1);
        } else {
            if (i == 0) start_evt_s.write(1);
        }

        // beat -> input_t (προσαρμόζεις αν αλλάξει το πλάτος)
        input_t pix;
        pix[0].range(15,0) = w.data;
        img_fifo.write(pix);
    }
}

static void run_core(hls::stream<input_t>  &img_fifo,
                     hls::stream<result_t> &res_fifo,
                     hls::stream<ap_uint<1>> &done_core_evt_s)
{
#pragma HLS INLINE off
    {{ project_name }}(img_fifo, res_fifo);
    // core-only done: υπάρχει αποτέλεσμα στο res_fifo
    done_core_evt_s.write(1);
}

static void egress_logits(hls::stream<result_t> &res_fifo,
                          hls::stream<axis_word_t> &m_axis,
                          hls::stream<ap_uint<1>> &done_e2e_evt_s)
{
#pragma HLS INLINE off
    result_t logits = res_fifo.read();

WRITE_LOGITS:
    for (unsigned c = 0; c < N_OUTPUT; ++c) {
#pragma HLS PIPELINE II=1
        axis_word_t w;
        w.data = logits[c].range(15,0);
        w.keep = 0x3;                           // 2 bytes valid
        w.last = (c == N_OUTPUT-1);
        m_axis.write(w);                        // μπλοκάρει σε TREADY

        if (w.last) {
            // end-to-end done: τελευταίο output handshake (περιλαμβάνει S2MM stalls)
            done_e2e_evt_s.write(1);
        }
    }
}

// ----------------- Παράλληλος μετρητής κύκλων (single-shot) -----------------
static void cycle_counter_proc(hls::stream<ap_uint<1>> &start_evt_s,
                               hls::stream<ap_uint<1>> &done_core_evt_s,
                               hls::stream<ap_uint<1>> &done_e2e_evt_s,
                               ap_uint<32> &cycles_core_out,
                               ap_uint<32> &cycles_e2e_out)
{
#pragma HLS INLINE off
    // 1) Περιμένουμε 1 start (blocking read)
    ap_uint<1> dummy;
    start_evt_s.read(dummy);

    // 2) Μετράμε μέχρι να έρθουν και τα δύο done
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

    // 3) Λάτσαρε στα outputs (μένουν μέχρι την επόμενη κλήση)
    cycles_core_out = cnt_core;
    cycles_e2e_out  = cnt_e2e;
}

// ============================ Top-level wrapper ==============================
void {{ project_name }}_stream(
    hls::stream<axis_word_t> &s_axis,
    hls::stream<axis_word_t> &m_axis,
#ifdef USE_AXIS_PERF
    hls::stream<perf_word_t> &m_axis_perf   // 2 beats: core, e2e (TLAST στο 2ο)
#else
    ap_uint<32> &cycles_core_out,           // -> AXI GPIO (input)
    ap_uint<32> &cycles_e2e_out             // -> AXI GPIO (input)
#endif
)
{
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return

#ifdef USE_AXIS_PERF
#pragma HLS INTERFACE axis port=m_axis_perf
#else
#pragma HLS INTERFACE ap_none port=cycles_core_out
#pragma HLS INTERFACE ap_none port=cycles_e2e_out
#endif

#pragma HLS DATAFLOW

    // Μικρά FIFOs (decoupling – όχι frame buffer)
    hls::stream<input_t>  img_fifo("img_fifo");
    hls::stream<result_t> res_fifo("res_fifo");
#pragma HLS STREAM variable=img_fifo depth=64
#pragma HLS STREAM variable=res_fifo depth=16

    // Event streams προς τον μετρητή
    hls::stream<ap_uint<1>> start_evt_s("start_evt_s");
    hls::stream<ap_uint<1>> done_core_evt_s("done_core_evt_s");
    hls::stream<ap_uint<1>> done_e2e_evt_s("done_e2e_evt_s");
#pragma HLS STREAM variable=start_evt_s     depth=2
#pragma HLS STREAM variable=done_core_evt_s depth=2
#pragma HLS STREAM variable=done_e2e_evt_s  depth=2

    // 1) ingest → 2) core → 3) egress
    ingest_pixels(s_axis, img_fifo, start_evt_s);
    run_core(img_fifo, res_fifo, done_core_evt_s);
    egress_logits(res_fifo, m_axis, done_e2e_evt_s);

#ifndef USE_AXIS_PERF
    // 4A) Με scalars (GPIO-friendly)
    cycle_counter_proc(start_evt_s, done_core_evt_s, done_e2e_evt_s,
                       cycles_core_out, cycles_e2e_out);
#else
    // 4B) Με AXIS perf side-channel (σταθερή cosim)
    ap_uint<32> cyc_core, cyc_e2e;
    cycle_counter_proc(start_evt_s, done_core_evt_s, done_e2e_evt_s,
                       cyc_core, cyc_e2e);

    perf_word_t pw;
    // beat 0: core-only cycles
    pw.data = cyc_core; pw.keep = -1; pw.strb = -1; pw.last = 0;
    m_axis_perf.write(pw);
    // beat 1: end-to-end cycles (TLAST)
    pw.data = cyc_e2e; pw.last = 1;
    m_axis_perf.write(pw);
#endif
}
