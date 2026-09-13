#include "radnet.h"
#include <math.h>
#include <stdlib.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static unsigned int xorshift32_local(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_uniform_local(unsigned int *state) {
    return (float)(xorshift32_local(state) >> 8) / (float)(1 << 24);
}


static float rand_gaussian_local(unsigned int *state) {
    float u1 = rand_uniform_local(state);
    float u2 = rand_uniform_local(state);
    if (u1 < 1e-12f) u1 = 1e-12f; 
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float complex rand_complex_gaussian_local(unsigned int *state) {
    float re = rand_gaussian_local(state);
    float im = rand_gaussian_local(state);
    return re + im * I;
}

static float complex rand_complex_small_local(unsigned int *state) {
    float mag = rand_uniform_local(state) * 0.1f;
    float phase = (rand_uniform_local(state) * 2.0f - 1.0f) * (float)M_PI;
    return mag * cexpf(I * phase);
}


void radnet_mutate_web(Web *web, float thresh_shift, float thresh_break,
                        float low_epsilon, float learning_rate, unsigned int *rng_state) {
    for (int s = 0; s < web->num_shells - 1; s++) {
        Weave *weave = &web->weaves[s];
        ForgetGate *fg = &web->forget_gates[s];
        Shell *src_shell = &web->shells[s];
        Shell *dst_shell = &web->shells[s + 1];
        int rows = weave->rows; 
        int cols = weave->cols;

        for (int r = 0; r < rows; r++) {
            float stress = dst_shell->stress[r];
            float complex e_dest = dst_shell->err_vec[r];
            
            for (int c = 0; c < cols; c++) {
                size_t idx = (size_t)r * (size_t)cols + (size_t)c;

                if (weave->frozen_mask[idx]) continue; 

                if (stress > thresh_break) {
                    
                    weave->links[idx] = 0.0f;
                    weave->stable_epochs[idx] = 0;

                    if (cols > 1) {
                        int new_c = c;
                        while (new_c == c) {
                            new_c = (int)(xorshift32_local(rng_state) % (unsigned int)cols);
                        }
                        size_t new_idx = (size_t)r * (size_t)cols + (size_t)new_c;
                        if (!weave->frozen_mask[new_idx]) {
                            weave->links[new_idx] = rand_complex_small_local(rng_state);
                            weave->stable_epochs[new_idx] = 0;
                        }
                    }
                } else if (stress > thresh_shift) {
                    
                    weave->links[idx] *= -1.0f;
                    weave->stable_epochs[idx] = 0;
                } else if (stress <= low_epsilon) {
                    
                    weave->stable_epochs[idx]++;
                    if (weave->stable_epochs[idx] >= 50) {
                        weave->frozen_mask[idx] = true;
                    }
                } else {
                    
                    float complex z_src = src_shell->z_state[c];
                    float complex delta_w = e_dest * conjf(z_src);
                    float complex noise = rand_complex_gaussian_local(rng_state);
                    float safe_stress = fminf(stress, 1.0f);
                    
                    weave->links[idx] = weave->links[idx] - (learning_rate * delta_w) + (sqrtf(safe_stress) * noise);
                    weave->stable_epochs[idx] = 0;
                }
            }
        }

        
        for (int r = 0; r < fg->n; r++) {
            float stress = dst_shell->stress[r];
            float complex e_dest = dst_shell->err_vec[r];
            
            for (int c = 0; c < fg->n; c++) {
                size_t idx = (size_t)r * (size_t)fg->n + (size_t)c;
                if (fg->frozen_mask[idx]) continue;

                if (stress > thresh_break) {
                    fg->weights[idx] = 0.0f;
                    fg->stable_epochs[idx] = 0;
                    if (fg->n > 1) {
                        int new_c = c;
                        while (new_c == c) {
                            new_c = (int)(xorshift32_local(rng_state) % (unsigned int)fg->n);
                        }
                        size_t new_idx = (size_t)r * (size_t)fg->n + (size_t)new_c;
                        if (!fg->frozen_mask[new_idx]) {
                            fg->weights[new_idx] = rand_complex_small_local(rng_state);
                            fg->stable_epochs[new_idx] = 0;
                        }
                    }
                } else if (stress > thresh_shift) {
                    fg->weights[idx] *= -1.0f;
                    fg->stable_epochs[idx] = 0;
                } else if (stress <= low_epsilon) {
                    fg->stable_epochs[idx]++;
                    if (fg->stable_epochs[idx] >= 50) {
                        fg->frozen_mask[idx] = true;
                    }
                } else {
                    float complex z_src = src_shell->z_state[c];
                    float complex delta_w = e_dest * conjf(z_src);
                    float complex noise = rand_complex_gaussian_local(rng_state);
                    float safe_stress = fminf(stress, 1.0f);
                    
                    fg->weights[idx] = fg->weights[idx] - (learning_rate * delta_w) + (sqrtf(safe_stress) * noise);
                    fg->stable_epochs[idx] = 0;
                }
            }

            
            if (stress > thresh_shift) {
                fg->bias[r] *= -1.0f;
            } else {
                float complex noise_b = rand_complex_gaussian_local(rng_state);
                float safe_stress = fminf(stress, 1.0f);
                fg->bias[r] = fg->bias[r] - (learning_rate * e_dest) + (sqrtf(safe_stress) * noise_b);
            }
        }
    }
}

void cylinder_mutate(Cylinder *cyl, float thresh_shift, float thresh_break,
                     float low_epsilon, float learning_rate, unsigned int *rng_state) {
    for (int w = 0; w < cyl->num_webs; w++) {
        radnet_mutate_web(&cyl->webs[w], thresh_shift, thresh_break, low_epsilon, learning_rate, rng_state);
    }
    
    int out_size = cyl->tradeoff.output_size;
    int in_size = cyl->tradeoff.input_size;
    float tradeoff_lr = learning_rate;
    for (int i = 0; i < out_size; i++) {
        float complex err = cyl->tradeoff.err_vec[i];
        cyl->tradeoff.bias[i] -= tradeoff_lr * crealf(err); 
        
        for (int j = 0; j < in_size; j++) {
            size_t idx = (size_t)i * (size_t)in_size + (size_t)j;
            if (cyl->tradeoff.frozen_mask[idx]) continue;
            
            float complex z_src = cyl->tradeoff.z_in[j];
            float complex delta_w = err * conjf(z_src);
            
            cyl->tradeoff.weights[idx] -= (tradeoff_lr * delta_w);
        }
    }
}

