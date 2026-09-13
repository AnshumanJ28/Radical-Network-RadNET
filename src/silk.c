#include "radnet.h"
#include <math.h>
#include <complex.h>


float complex silk_activate(float complex z, float gate_bias) {
    float mag = cabsf(z);
    float r_active = fmaxf(mag - gate_bias, 0.0f);
    float r_final = tanhf(r_active);

    if (mag == 0.0f) {
        
        return 0.0f;
    }

    float phase = cargf(z);
    return r_final * cexpf(I * phase);
}

void silk_activate_vec(const float complex *in, float complex *out,
                        int n, float gate_bias) {
    for (int i = 0; i < n; i++) {
        out[i] = silk_activate(in[i], gate_bias);
    }
}

