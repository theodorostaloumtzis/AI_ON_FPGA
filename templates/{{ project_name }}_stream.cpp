#include "{{ project_name }}_stream.h"   // axis_word_t + top prototype

#define N_INPUT   784   // 28×28
#define N_OUTPUT  10    // 10 classes

// ─────────────────────────────────────────────────────────────────────────────
// Start timing on TLAST of input (1) or on the FIRST input beat (0).
// Use 0 so the counter arms before the core can possibly finish.
#define START_ON_LAST_IN  0
// ─────────────────────────────────────────────────────────────────────────────

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

        // bit-copy beat -> input_t (προσαρμόζεις αν αλλάξει το πλάτος)
        input_t pix;
        pix[0].range(15,0) = w.data;
        img_fifo.write(pix);
    }
}

// When the core’s first logits are actually readable, emit 'done_core'.
// This timestamps the true “first availability” of output.
static void forward_and_mark(hls::stream<result_t> &core_out,
                             hls::stream<result_t> &res_fifo,
                             hls::stream<ap_uint<1>> &done_core_evt_s)
{
#pragma HLS INLINE off
    result_t logits = core_out.read(); // blocks until logits exist
    done_core_evt_s.write(1);          // core done at first readability
    res_fifo.write(logits);            // forward to egress
}

static void run_core(hls::stream<input_t>  &img_fifo,
                     hls::stream<result_t> &res_fifo,
                     hls::stream<ap_uint<1>> &done_core_evt_s)
{
#pragma HLS INLINE off

    hls::stream<result_t> core_out("core_out");
#pragma HLS STREAM variable=core_out depth=2

#pragma HLS DATAFLOW
    {{ project_name }}(img_fifo, core_out);
    forward_and_mark(core_out, res_fifo, done_core_evt_s);
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
// Περιμένει 1 start, αδειάζει τυχόν "μπαγιάτικα" done, και μετράει μέχρι να
// λάβει done_core ΚΑΙ done_e2e. Το μέτρημα είναι “exclusive” του κύκλου done
// (ελέγχει πρώτα για done, μετά αυξάνει).
static void cycle_counter_proc(hls::stream<ap_uint<1>> &start_evt_s,
                               hls::stream<ap_uint<1>> &done_core_evt_s,
                               hls::stream<ap_uint<1>> &done_e2e_evt_s,
                               ap_uint<32> &cycles_core_out,
                               ap_uint<32> &cycles_e2e_out)
{
#pragma HLS INLINE off

    // 1) Περιμένουμε 1 start (blocking read — ασφαλές: το ingest το γράφει πάντα)
    ap_uint<1> dummy;
    start_evt_s.read(dummy);

    // 2) Άδειασε τυχόν παλιά done pulses (από προηγούμενο frame)
FLUSH_CORE:
    while (!done_core_evt_s.empty())  { (void)done_core_evt_s.read(); }
FLUSH_E2E:
    while (!done_e2e_evt_s.empty())   { (void)done_e2e_evt_s.read(); }

    ap_uint<32> cnt_core = 0, cnt_e2e = 0;
    bool run_core = true, run_e2e = true;

COUNT_LOOP:
    while (run_core || run_e2e) {
#pragma HLS PIPELINE II=1

        // Έλεγχος για ολοκλήρωση ΠΡΙΝ το increase — ώστε να μη μετρήσουμε extra κύκλο
        if (run_core && !done_core_evt_s.empty()) {
            (void)done_core_evt_s.read();
            run_core = false;
        }
        if (run_e2e && !done_e2e_evt_s.empty()) {
            (void)done_e2e_evt_s.read();
            run_e2e = false;
        }

        if (run_core) cnt_core++;
        if (run_e2e)  cnt_e2e++;
    }

    // 3) Λάτσαρε στα outputs (κρατούνται μέχρι την επόμενη κλήση/καρέ)
    cycles_core_out = cnt_core;
    cycles_e2e_out  = cnt_e2e;
}

// ============================ Top-level wrapper ==============================
void {{ project_name }}_stream(
    hls::stream<axis_word_t> &s_axis,
    hls::stream<axis_word_t> &m_axis,
    ap_uint<32> &cycles_core_out,   // -> AXI GPIO channel 1 (input)
    ap_uint<32> &cycles_e2e_out     // -> AXI GPIO channel 2 (input)
)
{
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return

// ap_none για απευθείας καλώδια προς GPIO
#pragma HLS INTERFACE ap_none port=cycles_core_out
#pragma HLS INTERFACE ap_none port=cycles_e2e_out

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
#pragma HLS STREAM variable=start_evt_s     depth=4
#pragma HLS STREAM variable=done_core_evt_s depth=4
#pragma HLS STREAM variable=done_e2e_evt_s  depth=4

    // 1) ingest → 2) core (+ forward mark) → 3) egress
    ingest_pixels(s_axis, img_fifo, start_evt_s);
    run_core(img_fifo, res_fifo, done_core_evt_s);
    egress_logits(res_fifo, m_axis, done_e2e_evt_s);

    // 4) Παράλληλος μετρητής
    cycle_counter_proc(start_evt_s, done_core_evt_s, done_e2e_evt_s,
                       cycles_core_out, cycles_e2e_out);
}
