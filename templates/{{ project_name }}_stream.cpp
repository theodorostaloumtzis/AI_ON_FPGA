


#include "{{ project_name }}_stream.h"   // axis_word_t + top prototype

#define N_INPUT 784 // 28×28 pixels
#define N_OUTPUT 10 // 10 classes

// Top‑level wrapper ----------------------------------------------------------
void {{ project_name }}_stream(
    hls::stream<axis_word_t> &s_axis,
    hls::stream<axis_word_t> &m_axis)
{
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS aggregate variable=s_axis
#pragma HLS aggregate variable=m_axis


    const unsigned NPIX   = N_INPUT; // 28×28 = 784
    const unsigned NCLASS = N_OUTPUT; // 10

    // Internal FIFOs with CNN‑native datatypes ------------------------------
    hls::stream<input_t>  img_fifo("img_fifo");
    hls::stream<result_t> res_fifo("res_fifo");
#pragma HLS STREAM variable=img_fifo depth=784
#pragma HLS STREAM variable=res_fifo depth=32

    // 1. Ingest pixels -------------------------------------------------------
    PIXEL_LOOP: for (unsigned i = 0; i < NPIX; ++i) {
#pragma HLS PIPELINE II=1
        axis_word_t word = s_axis.read();
        input_t pix;
        pix[0].range(15,0) = word.data;   // pure bit‑copy into Q6.10
        img_fifo.write(pix);
    }

    // 2. Run network (fully inlined) ----------------------------------------
    {{ project_name }}(img_fifo, res_fifo);

    // 3. Stream logits out ---------------------------------------------------
    result_t logits = res_fifo.read();
    LOGIT_LOOP: for (unsigned c = 0; c < NCLASS; ++c) {
#pragma HLS PIPELINE II=1
        axis_word_t word;
        word.data = logits[c].range(15,0);
        word.keep = 0x3;                   // both bytes valid
        word.last = (c == NCLASS-1);
        m_axis.write(word);
    }
}
