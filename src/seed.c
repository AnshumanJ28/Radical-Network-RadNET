#include "radnet.h"
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>



typedef struct {
    float dist;   
    int index;    
} FreqEntry;

static int freq_cmp(const void *a, const void *b) {
    const FreqEntry *fa = (const FreqEntry *)a;
    const FreqEntry *fb = (const FreqEntry *)b;
    if (fa->dist < fb->dist) return -1;
    if (fa->dist > fb->dist) return 1;
    return 0;
}

int radnet_seed_from_image(const float *pixels, int width, int height,
                            Web *web) {
    if (!pixels || width <= 0 || height <= 0 || !web) return -1;

    
    long total_nodes = 0;
    for (int s = 1; s <= web->num_shells; s++) {
        total_nodes += radnet_shell_node_count(s);
    }

    long num_pixels = (long)width * (long)height;
    if (num_pixels < total_nodes) {
        
        return -1;
    }

    
    float complex *in = malloc(sizeof(float complex) * (size_t)num_pixels);
    float complex *out = malloc(sizeof(float complex) * (size_t)num_pixels);
    if (!in || !out) {
        free(in); free(out);
        return -1;
    }

    for (long i = 0; i < num_pixels; i++) {
        in[i] = pixels[i] + 0.0f * I;
    }

    
    float complex *temp = malloc(sizeof(float complex) * (size_t)num_pixels);
    
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float complex sum = 0.0f + 0.0f * I;
            for (int u = 0; u < width; u++) {
                float angle = -2.0f * 3.1415926535f * (float)(x * u) / (float)width;
                sum += in[y * width + u] * cexpf(I * angle);
            }
            temp[y * width + x] = sum;
        }
    }
    
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float complex sum = 0.0f + 0.0f * I;
            for (int v = 0; v < height; v++) {
                float angle = -2.0f * 3.1415926535f * (float)(y * v) / (float)height;
                sum += temp[v * width + x] * cexpf(I * angle);
            }
            out[y * width + x] = sum;
        }
    }
    free(temp);
    free(in);

    
    float norm = 1.0f / (float)num_pixels;
    for (long i = 0; i < num_pixels; i++) {
        out[i] *= norm;
    }

    
    FreqEntry *freqs = malloc(sizeof(FreqEntry) * (size_t)num_pixels);
    if (!freqs) {
        free(out);
        return -1;
    }

    for (int v = 0; v < height; v++) {
        int dv = (v <= height / 2) ? v : (v - height);
        for (int u = 0; u < width; u++) {
            int du = (u <= width / 2) ? u : (u - width);
            long idx = (long)v * width + u;
            freqs[idx].dist = sqrtf((float)(du * du + dv * dv));
            freqs[idx].index = (int)idx;
        }
    }
    qsort(freqs, (size_t)num_pixels, sizeof(FreqEntry), freq_cmp);

    
    long cursor = 0;
    for (int s = 0; s < web->num_shells; s++) {
        Shell *shell = &web->shells[s];
        for (int i = 0; i < shell->num_nodes; i++) {
            long fidx = freqs[cursor + i].index;
            shell->seed[i] = out[fidx];
            shell->z_state[i] = out[fidx];
        }
        cursor += shell->num_nodes;
    }

    free(freqs);
    free(out);
    return 0;
}

