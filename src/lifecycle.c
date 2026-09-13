#include "radnet.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_uniform(unsigned int *state) {
    
    return (float)(xorshift32(state) >> 8) / (float)(1 << 24);
}

static float complex rand_complex_small(unsigned int *state) {
    float mag = rand_uniform(state) * 0.57f;
    float phase = (rand_uniform(state) * 2.0f - 1.0f) * (float)M_PI;
    return mag * cexpf(I * phase);
}



int shell_init(Shell *sh, int num_nodes) {
    sh->num_nodes = num_nodes;
    sh->z_state = calloc((size_t)num_nodes, sizeof(float complex));
    sh->seed    = calloc((size_t)num_nodes, sizeof(float complex));
    sh->err_vec = calloc((size_t)num_nodes, sizeof(float complex));
    sh->gates   = calloc((size_t)num_nodes, sizeof(float));
    sh->stress  = calloc((size_t)num_nodes, sizeof(float));
    if (!sh->z_state || !sh->seed || !sh->err_vec || !sh->gates || !sh->stress) {
        shell_free(sh);
        return -1;
    }
    return 0;
}

void shell_free(Shell *sh) {
    free(sh->z_state); sh->z_state = NULL;
    free(sh->seed);    sh->seed    = NULL;
    free(sh->err_vec); sh->err_vec = NULL;
    free(sh->gates);   sh->gates   = NULL;
    free(sh->stress);  sh->stress  = NULL;
    sh->num_nodes = 0;
}



int weave_init(Weave *w, int rows, int cols, unsigned int *rng_state) {
    w->rows = rows;
    w->cols = cols;
    size_t n = (size_t)rows * (size_t)cols;

    w->links         = calloc(n, sizeof(float complex));
    w->bias          = calloc((size_t)rows, sizeof(float complex));
    w->frozen_mask   = calloc(n, sizeof(bool));
    w->stable_epochs = calloc(n, sizeof(int));
    if (!w->links || !w->bias || !w->frozen_mask || !w->stable_epochs) {
        weave_free(w);
        return -1;
    }

    for (int r = 0; r < rows; r++) {
        
        for (int c = 0; c < cols; c++) {
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            w->links[idx] = 0.0f + 0.0f * I;
            w->frozen_mask[idx] = true;
            w->stable_epochs[idx] = 0;
        }

        
        int k = (cols < 3) ? cols : 3;
        int selected = 0;
        while (selected < k) {
            int c = (int)(xorshift32(rng_state) % (unsigned int)cols);
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            if (w->frozen_mask[idx] == true) {
                w->links[idx] = rand_complex_small(rng_state);
                w->frozen_mask[idx] = false;
                selected++;
            }
        }
    }
    for (int i = 0; i < rows; i++) {
        w->bias[i] = rand_complex_small(rng_state);
    }
    return 0;
}

void weave_free(Weave *w) {
    free(w->links);         w->links = NULL;
    free(w->bias);          w->bias = NULL;
    free(w->frozen_mask);   w->frozen_mask = NULL;
    free(w->stable_epochs); w->stable_epochs = NULL;
    w->rows = w->cols = 0;
}



int forget_gate_init(ForgetGate *fg, int n, unsigned int *rng_state) {
    fg->n = n;
    fg->weights = calloc((size_t)n * (size_t)n, sizeof(float complex));
    fg->bias    = calloc((size_t)n, sizeof(float complex));
    fg->frozen_mask = calloc((size_t)n * (size_t)n, sizeof(bool));
    fg->stable_epochs = calloc((size_t)n * (size_t)n, sizeof(int));
    if (!fg->weights || !fg->bias || !fg->frozen_mask || !fg->stable_epochs) {
        forget_gate_free(fg);
        return -1;
    }
    for (size_t i = 0; i < (size_t)n * (size_t)n; i++) {
        fg->weights[i] = rand_complex_small(rng_state);
        fg->frozen_mask[i] = false;
        fg->stable_epochs[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        fg->bias[i] = rand_complex_small(rng_state);
    }
    return 0;
}

void forget_gate_free(ForgetGate *fg) {
    free(fg->weights); fg->weights = NULL;
    free(fg->bias);    fg->bias = NULL;
    free(fg->frozen_mask); fg->frozen_mask = NULL;
    free(fg->stable_epochs); fg->stable_epochs = NULL;
    fg->n = 0;
}



int web_init(Web *web, int num_shells, unsigned int *rng_state) {
    if (num_shells < 2) return -1;
    web->num_shells = num_shells;
    web->shells = calloc(num_shells, sizeof(Shell));
    web->weaves = calloc(num_shells - 1, sizeof(Weave));
    web->forget_gates = calloc(num_shells - 1, sizeof(ForgetGate));
    if (!web->shells || !web->weaves || !web->forget_gates) {
        web_free(web);
        return -1;
    }

    int *sizes = calloc(num_shells, sizeof(int));
    if (!sizes) {
        web_free(web);
        return -1;
    }
    for (int s = 0; s < num_shells; s++) {
        sizes[s] = radnet_shell_node_count(s + 1);
    }

    for (int s = 0; s < num_shells; s++) {
        if (shell_init(&web->shells[s], sizes[s]) != 0) {
            for (int k = 0; k < s; k++) shell_free(&web->shells[k]);
            free(sizes);
            return -1;
        }
    }

    for (int s = 0; s < num_shells - 1; s++) {
        int rows = sizes[s + 1]; 
        int cols = sizes[s];     
        if (weave_init(&web->weaves[s], rows, cols, rng_state) != 0) {
            for (int k = 0; k < s; k++) {
                weave_free(&web->weaves[k]);
                forget_gate_free(&web->forget_gates[k]);
            }
            for (int k = 0; k < num_shells; k++) shell_free(&web->shells[k]);
            free(sizes);
            return -1;
        }
        if (forget_gate_init(&web->forget_gates[s], cols, rng_state) != 0) {
            weave_free(&web->weaves[s]);
            for (int k = 0; k < s; k++) {
                weave_free(&web->weaves[k]);
                forget_gate_free(&web->forget_gates[k]);
            }
            for (int k = 0; k < num_shells; k++) shell_free(&web->shells[k]);
            free(sizes);
            return -1;
        }
    }
    free(sizes);
    return 0;
}

void web_free(Web *web) {
    if (web->shells) {
        for (int s = 0; s < web->num_shells; s++) shell_free(&web->shells[s]);
        free(web->shells); web->shells = NULL;
    }
    if (web->weaves) {
        for (int s = 0; s < web->num_shells - 1; s++) {
            weave_free(&web->weaves[s]);
            forget_gate_free(&web->forget_gates[s]);
        }
        free(web->weaves); web->weaves = NULL;
        free(web->forget_gates); web->forget_gates = NULL;
    }
    web->num_shells = 0;
}



int tradeoff_layer_init(TradeoffLayer *tl, int in_size, int out_size, unsigned int *rng_state) {
    tl->input_size = in_size;
    tl->output_size = out_size;
    size_t n = (size_t)in_size * (size_t)out_size;
    tl->weights = calloc(n, sizeof(float complex));
    tl->bias = calloc((size_t)out_size, sizeof(float)); 
    tl->frozen_mask = calloc(n, sizeof(bool));
    tl->stable_epochs = calloc(n, sizeof(int));
    tl->err_vec = calloc((size_t)out_size, sizeof(float complex));
    tl->z_in = calloc((size_t)in_size, sizeof(float complex));
    tl->z_out = calloc((size_t)out_size, sizeof(float complex));
    if (!tl->weights || !tl->bias || !tl->frozen_mask || !tl->stable_epochs || !tl->err_vec || !tl->z_in || !tl->z_out) {
        tradeoff_layer_free(tl);
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        tl->weights[i] = rand_complex_small(rng_state);
        tl->frozen_mask[i] = false;
        tl->stable_epochs[i] = 0;
    }
    for (int i = 0; i < out_size; i++) {
        tl->bias[i] = rand_uniform(rng_state) * 0.1f; 
    }
    return 0;
}

void tradeoff_layer_free(TradeoffLayer *tl) {
    free(tl->weights); tl->weights = NULL;
    free(tl->bias); tl->bias = NULL;
    free(tl->frozen_mask); tl->frozen_mask = NULL;
    free(tl->stable_epochs); tl->stable_epochs = NULL;
    free(tl->err_vec); tl->err_vec = NULL;
    free(tl->z_in); tl->z_in = NULL;
    free(tl->z_out); tl->z_out = NULL;
    tl->input_size = 0;
    tl->output_size = 0;
}



int cylinder_init(Cylinder *cyl, int num_webs, int num_shells, int out_classes, unsigned int *rng_state) {
    cyl->num_webs = num_webs;
    cyl->webs = calloc((size_t)num_webs, sizeof(Web));
    if (!cyl->webs) return -1;

    for (int w = 0; w < num_webs; w++) {
        if (web_init(&cyl->webs[w], num_shells, rng_state) != 0) {
            for (int k = 0; k < w; k++) web_free(&cyl->webs[k]);
            free(cyl->webs);
            cyl->webs = NULL;
            return -1;
        }
    }
    
    int final_shell_size = radnet_shell_node_count(num_shells);
    int tradeoff_in_size = num_webs * final_shell_size;
    
    if (tradeoff_layer_init(&cyl->tradeoff, tradeoff_in_size, out_classes, rng_state) != 0) {
        for (int w = 0; w < num_webs; w++) web_free(&cyl->webs[w]);
        free(cyl->webs);
        cyl->webs = NULL;
        return -1;
    }
    
    return 0;
}

void cylinder_free(Cylinder *cyl) {
    if (!cyl->webs) return;
    for (int w = 0; w < cyl->num_webs; w++) web_free(&cyl->webs[w]);
    free(cyl->webs);
    cyl->webs = NULL;
    cyl->num_webs = 0;
    tradeoff_layer_free(&cyl->tradeoff);
}

