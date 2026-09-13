#include "radnet.h"
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>


void radnet_compute_echo(Web *web, int label, float lambda) {
    
    
    for (int s = web->num_shells - 2; s >= 0; s--) {
        Weave *weave = &web->weaves[s];
        Shell *src_shell = &web->shells[s];
        Shell *dst_shell = &web->shells[s + 1];

        memset(src_shell->err_vec, 0, weave->cols * sizeof(float complex));
        radnet_cgemv_conj(weave->rows, weave->cols, weave->links, dst_shell->err_vec, src_shell->err_vec);
    }

    
    for (int shell_index = 0; shell_index < web->num_shells; shell_index++) {
        int s = shell_index + 1;
        float falloff = expf(-lambda * (float)(web->num_shells - s));
        Shell *shell = &web->shells[shell_index];
        for (int j = 0; j < shell->num_nodes; j++) {
            float err_mag = cabsf(shell->err_vec[j]);
            float z_mag = cabsf(shell->z_state[j]);
            shell->stress[j] = err_mag * z_mag * falloff;
        }
    }
}


void cylinder_echo(Cylinder *cyl, const float *out_probs, int label, float lambda) {
    if (cyl->num_webs <= 0 || !cyl->webs) return;
    
    int out_size = cyl->tradeoff.output_size;
    float complex* d_logits = cyl->tradeoff.err_vec;
    
    
    for (int i = 0; i < out_size; i++) {
        
        float label_prob = (i == label) ? 1.0f : 0.0f;
        float error_mag = out_probs[i] - label_prob;
        
        
        
        d_logits[i] = error_mag + 0.0f * I;
        cyl->tradeoff.err_vec[i] = d_logits[i];
    }

    
    int in_size = cyl->tradeoff.input_size;
    float complex* d_in = calloc((size_t)in_size, sizeof(float complex));
    
    memset(d_in, 0, in_size * sizeof(float complex));
    radnet_cgemv_conj(out_size, in_size, cyl->tradeoff.weights, d_logits, d_in);
                
    
    int final_shell_size = cyl->webs[0].shells[cyl->webs[0].num_shells - 1].num_nodes;
    
    for (int w = 0; w < cyl->num_webs; w++) {
        Web* web = &cyl->webs[w];
        Shell* final_shell = &web->shells[web->num_shells - 1];
        
        for (int i = 0; i < final_shell_size; i++) {
            final_shell->err_vec[i] = d_in[w * final_shell_size + i];
        }
        
        
        radnet_compute_echo(web, label, lambda);
    }
    
    free(d_in);
}

