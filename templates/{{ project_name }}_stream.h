#ifndef {{ project_name }}_STREAM_H_
#define {{ project_name }}_STREAM_H_

#include <ap_axi_sdata.h>   // provides ap_axiu + AXIS_ENABLE_* flags
#include <hls_stream.h>
#include "{{ project_name }}.h"          // generated network core
#include "defines.h"

// 16-bit data   + KEEP + LAST   (no STRB / USER / ID / DEST)
typedef ap_axiu<16, 0, 0, 0,(AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST)>  // EnableSignals
        axis_word_t;

// --------------------------- Function proto --------------------------------
void {{ project_name }}_stream(
        hls::stream<axis_word_t> &s_axis,
        hls::stream<axis_word_t> &m_axis);

#endif // {{ project_name }}_STREAM_H_
