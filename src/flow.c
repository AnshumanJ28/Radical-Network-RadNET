#include "radnet.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

void radnet_cgemv(int rows, int cols, const float complex* A, const float complex* x, float complex* y) {
    for (int i = 0; i < rows; i++) {
        float complex acc = 0.0f + 0.0f * I;
        const float complex* row = &A[i * cols];
        for (int j = 0; j < cols; j++) {
            acc += row[j] * x[j];
        }
        y[i] += acc;
    }
}

void radnet_cgemv_conj(int rows, int cols, const float complex* A, const float complex* x, float complex* y) {
    for (int j = 0; j < cols; j++) {
        float complex acc = 0.0f + 0.0f * I;
        for (int i = 0; i < rows; i++) {
            acc += conjf(A[i * cols + j]) * x[i];
        }
        y[j] += acc;
    }
}


static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}


static void forget_gate_matvec(const ForgetGate *fg, const float complex *z,
                                float complex *out) {
    int n = fg->n;
    for (int i = 0; i < n; i++) {
        float complex acc = fg->bias[i];
        const float complex *row = &fg->weights[(size_t)i * (size_t)n];
        for (int j = 0; j < n; j++) {
            acc += row[j] * z[j];
        }
        out[i] = acc;
    }
}


int radnet_forward_shell(Web *web, int s, float silk_gate_bias) {
    if (s < 0 || s >= web->num_shells - 1) return -1;

    Shell *src = &web->shells[s];
    Shell *dst = &web->shells[s + 1];
    Weave *weave = &web->weaves[s];
    ForgetGate *fg = &web->forget_gates[s];

    if (fg->n != src->num_nodes) return -1;
    if (weave->cols != src->num_nodes) return -1;
    if (weave->rows != dst->num_nodes) return -1;

    int n_src = src->num_nodes;
    int n_dst = dst->num_nodes;

    
    float complex *gate_pre = malloc((size_t)n_src * sizeof(float complex));
    float *gate_mask = malloc((size_t)n_src * sizeof(float));
    float complex *gated = malloc((size_t)n_src * sizeof(float complex));
    if (!gate_pre || !gate_mask || !gated) {
        free(gate_pre); free(gate_mask); free(gated);
        return -1;
    }

    forget_gate_matvec(fg, src->z_state, gate_pre);
    for (int i = 0; i < n_src; i++) {
        gate_mask[i] = sigmoidf(cabsf(gate_pre[i]));
        src->gates[i] = gate_mask[i]; 
        gated[i] = gate_mask[i] * src->z_state[i]; 
        if (src->seed != NULL) {
            gated[i] += src->seed[i];
        }
    }

    
    float complex *y = malloc((size_t)n_dst * sizeof(float complex));
    if (!y) {
        free(gate_pre); free(gate_mask); free(gated);
        return -1;
    }

    
    for (int i = 0; i < n_dst; i++) {
        y[i] = weave->bias[i];
    }

    radnet_cgemv(n_dst, n_src, weave->links, gated, y);

    
    silk_activate_vec(y, dst->z_state, n_dst, silk_gate_bias);

    free(gate_pre);
    free(gate_mask);
    free(gated);
    free(y);
    return 0;
}


int cylinder_forward(Cylinder *cyl, float *out_probs) {
    if (cyl->num_webs <= 0 || !cyl->webs) return -1;
    
    int num_shells = cyl->webs[0].num_shells;
    int final_shell_size = cyl->webs[0].shells[num_shells - 1].num_nodes;
    int tradeoff_in_size = cyl->num_webs * final_shell_size;
    
    if (cyl->tradeoff.input_size != tradeoff_in_size) return -1;
    
    float complex* concat_in = cyl->tradeoff.z_in;
    
    
    for (int w = 0; w < cyl->num_webs; w++) {
        Shell* final_shell = &cyl->webs[w].shells[num_shells - 1];
        memcpy(&concat_in[w * final_shell_size], final_shell->z_state, final_shell_size * sizeof(float complex));
    }
    
    int out_size = cyl->tradeoff.output_size;
    float complex* complex_out = cyl->tradeoff.z_out;
    
    
    memset(complex_out, 0, out_size * sizeof(float complex));
    radnet_cgemv(out_size, tradeoff_in_size, cyl->tradeoff.weights, concat_in, complex_out);
                
    
    float max_val = -1e9f;
    for (int i = 0; i < out_size; i++) {
        float real_val = crealf(complex_out[i]);
        out_probs[i] = real_val + cyl->tradeoff.bias[i];
        if (out_probs[i] > max_val) max_val = out_probs[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < out_size; i++) {
        out_probs[i] = expf(out_probs[i] - max_val); 
        sum_exp += out_probs[i];
    }
    for (int i = 0; i < out_size; i++) {
        out_probs[i] /= sum_exp;
    }
    
    return 0;
}

