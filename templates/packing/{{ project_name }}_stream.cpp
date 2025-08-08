#include "{{ project_name }}_stream.h"   // axis64_t + axis16_t + top prototype

#define N_INPUT    784   // total pixels = 28×28
#define N_OUTPUT    10   // number of classes
#define PACK_SIZE    4   // pixels per 64-bit beat
#define N_PKT    (N_INPUT / PACK_SIZE)  // 196 beats

// Top-level wrapper ----------------------------------------------------------
void {{ project_name }}_stream(
    hls::stream<axis64_t> &s_axis,
    hls::stream<axis16_t> &m_axis)
{
#pragma HLS INTERFACE axis         port=s_axis
#pragma HLS INTERFACE axis         port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS aggregate variable=s_axis
#pragma HLS aggregate variable=m_axis

    // Internal FIFOs ---------------------------------------------------------
    hls::stream<input_t>  img_fifo("img_fifo");
    hls::stream<result_t> res_fifo("res_fifo");
#pragma HLS STREAM variable=img_fifo depth=N_INPUT  // can hold all 784 pixels
#pragma HLS STREAM variable=res_fifo depth=N_OUTPUT

    // 1. Ingest N_PKT × 64-bit words, unpack 4 pixels each ---------------
    PIXEL_LOOP: for (unsigned pkt = 0; pkt < N_PKT; ++pkt) {
    #pragma HLS PIPELINE II=1
        axis64_t aw = s_axis.read();
        UNPACK_PIXELS: for (int j = 0; j < PACK_SIZE; ++j) {
        #pragma HLS UNROLL
            input_t pix;
            pix[0].range(15,0) = aw.data.range(16*j + 15, 16*j);
            img_fifo.write(pix);
        }
    }

    // 2. Run network (fully inlined) ----------------------------------------
    {{ project_name }}(img_fifo, res_fifo);

    // 3. Stream one 16-bit logit per beat out -------------------------------
    result_t logits = res_fifo.read();
    LOGIT_LOOP: for (unsigned c = 0; c < N_OUTPUT; ++c) {
    #pragma HLS PIPELINE II=1
        axis16_t ow;
        ow.data = logits[c].range(15,0);
        ow.keep = 0x3;                  // both bytes valid
        ow.last = (c == N_OUTPUT-1);    // LAST on final logit
        m_axis.write(ow);
    }
}
