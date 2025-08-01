#ifndef {{ project_name }}_STREAM_H_
#define {{ project_name }}_STREAM_H_

#include <ap_axi_sdata.h>   // provides ap_axiu + AXIS_ENABLE_* flags
#include <hls_stream.h>
#include "{{ project_name }}.h"          // generated network core
#include "defines.h"

// 64-bit data   + KEEP + LAST   (no STRB / USER / ID / DEST)
typedef ap_axiu<64, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST)> axis64_t;
// 16-bit data   + KEEP + LAST   (no STRB / USER / ID / DEST)   
typedef ap_axiu<16, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST)> axis16_t;


// --------------------------- Function proto --------------------------------
void {{ project_name }}_stream(
        hls::stream<axis64_t> &s_axis,
        hls::stream<axis16_t> &m_axis);

#endif // {{ project_name }}_STREAM_H_
