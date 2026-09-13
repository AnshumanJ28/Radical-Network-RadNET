#include "radnet.h"
#include <math.h>
#include <stdlib.h>
#include <complex.h>


int radnet_compute_snag(const Web *web, int label, float *loss_out) {
    const Shell *final_shell = &web->shells[web->num_shells - 1];
    int n = final_shell->num_nodes;
    if (label < 0 || label >= n || !loss_out) return -1;

    float *mags = malloc((size_t)n * sizeof(float));
    if (!mags) return -1;

    float max_mag = -INFINITY;
    for (int i = 0; i < n; i++) {
        mags[i] = cabsf(final_shell->z_state[i]);
        if (mags[i] > max_mag) max_mag = mags[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(mags[i] - max_mag);
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    float p_label = 0.0f;
    p_label = expf(mags[label] - max_mag) / sum_exp;
#pragma GCC diagnostic pop
    
    if (p_label < 1e-12f) p_label = 1e-12f;

    *loss_out = -logf(p_label);

    free(mags);
    return 0;
}

