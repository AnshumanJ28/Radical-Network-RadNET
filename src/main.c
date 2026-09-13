#include "radnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>



static void test_silk(void) {
    printf("\n-- Silk activation sanity check --\n");

    
    float complex z1 = 0.3f + 0.1f * I; 
    float complex out1 = silk_activate(z1, 0.5f);
    printf("z1=%.3f%+.3fi |z1|=%.3f -> Silk=%.3f%+.3fi (expect ~0)\n",
           crealf(z1), cimagf(z1), cabsf(z1), crealf(out1), cimagf(out1));

    
    float complex z2 = 3.0f + 4.0f * I; 
    float complex out2 = silk_activate(z2, 1.0f);
    printf("z2=%.3f%+.3fi |z2|=%.3f arg=%.3f -> Silk=%.3f%+.3fi |out|=%.3f arg=%.3f\n",
           crealf(z2), cimagf(z2), cabsf(z2), cargf(z2),
           crealf(out2), cimagf(out2), cabsf(out2), cargf(out2));
}

static void test_forward_pass(void) {
    printf("\n-- Domain 3: BLAS Forward Flow test (full 15-transition pass) --\n");

    unsigned int rng_state = 999u;
    Web web;
    if (web_init(&web, 16, &rng_state) != 0) {
        fprintf(stderr, "web_init FAILED\n");
        return;
    }

    
    for (int i = 0; i < web.shells[0].num_nodes; i++) {
        float mag = 1.0f + 0.5f * (float)i;
        float phase = 0.3f * (float)i;
        web.shells[0].seed[i] = mag * cexpf(I * phase);
        web.shells[0].z_state[i] = mag * cexpf(I * phase);
    }

    const float silk_gate_bias = 0.05f;

    for (int s = 0; s < 16 - 1; s++) {
        if (radnet_forward_shell(&web, s, silk_gate_bias) != 0) {
            fprintf(stderr, "radnet_forward_shell FAILED at shell %d\n", s);
            web_free(&web);
            return;
        }
    }

    printf("Forward pass completed Shell 1 -> Shell 16 with no errors.\n");
    printf("Shell 16 (final) has %d nodes. Sample outputs:\n", web.shells[15].num_nodes);
    for (int i = 0; i < 5 && i < web.shells[15].num_nodes; i++) {
        float complex z = web.shells[15].z_state[i];
        printf("  node[%d] = %.5f%+.5fi  |z|=%.5f arg=%.5f\n",
               i, crealf(z), cimagf(z), cabsf(z), cargf(z));
    }

    
    int bad = 0;
    for (int s = 1; s < 16; s++) { 
        for (int i = 0; i < web.shells[s].num_nodes; i++) {
            float complex z = web.shells[s].z_state[i];
            float m = cabsf(z);
            if (isnan(m) || isinf(m) || m > 1.0001f) {
                printf("  ANOMALY shell=%d node=%d |z|=%f\n", s, i, m);
                bad++;
            }
        }
    }
    printf(bad == 0
        ? "All shell magnitudes finite and <= 1.0 (Silk elastic limit holds). PASS.\n"
        : "FAILURES detected above.\n");

    web_free(&web);
}

static void test_seed_and_full_pipeline(void) {
    printf("\n-- Domain 1 (FFTW Seed) + Domain 4 (Echo/Mutation) full pipeline --\n");

    unsigned int rng_state = 42u;
    Web web;
    if (web_init(&web, 16, &rng_state) != 0) {
        fprintf(stderr, "web_init FAILED\n");
        return;
    }

    
    const int W = 32, H = 32;
    float *image = malloc((size_t)(W * H) * sizeof(float));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float gx = (float)x / (float)W;
            float gy = (float)y / (float)H;
            image[y * W + x] = gx + gy + 0.05f * sinf(10.0f * gx) * cosf(10.0f * gy);
        }
    }

    if (radnet_seed_from_image(image, W, H, &web) != 0) {
        fprintf(stderr, "radnet_seed_from_image FAILED\n");
        free(image);
        web_free(&web);
        return;
    }
    free(image);

    printf("Seed injected. Shell 1 (DC/low-freq) sample magnitudes:\n");
    for (int i = 0; i < web.shells[0].num_nodes; i++) {
        printf("  shell1 node[%d] |z|=%.4f\n", i, cabsf(web.shells[0].seed[i]));
    }
    printf("Shell 16 (high-freq) sample magnitudes (first 5 of %d):\n",
           web.shells[15].num_nodes);
    for (int i = 0; i < 5; i++) {
        printf("  shell16 node[%d] |z|=%.4f\n", i, cabsf(web.shells[15].seed[i]));
    }
    printf("(For a smooth image, shell1's DC term should dominate in magnitude\n");
    printf(" over typical shell16 high-frequency bins -- confirms low/high freq\n");
    printf(" routed to inner/outer shells as intended.)\n");

    
    const int label = 3;
    const float silk_gate_bias = 0.05f;
    const float lambda = 0.15f;
    const float thresh_shift = 0.6f;
    const float thresh_break = 1.2f;
    const float low_epsilon = 0.01f;
    const float learning_rate = 0.01f;
    const int epochs = 5;

    printf("\nRunning %d full epochs (forward -> Snag -> Echo -> Mutate):\n", epochs);
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int s = 0; s < 16 - 1; s++) {
            if (radnet_forward_shell(&web, s, silk_gate_bias) != 0) {
                fprintf(stderr, "forward FAILED epoch %d shell %d\n", epoch, s);
                web_free(&web);
                return;
            }
        }

        float loss = 0.0f;
        if (radnet_compute_snag(&web, label, &loss) != 0) {
            fprintf(stderr, "Snag FAILED epoch %d\n", epoch);
            web_free(&web);
            return;
        }

        radnet_compute_echo(&web, label, lambda);
        radnet_mutate_web(&web, thresh_shift, thresh_break, low_epsilon, learning_rate, &rng_state);

        
        float inner_stress_sum = 0.0f, outer_stress_sum = 0.0f;
        for (int i = 0; i < web.shells[0].num_nodes; i++) inner_stress_sum += web.shells[0].stress[i];
        for (int i = 0; i < web.shells[15].num_nodes; i++) outer_stress_sum += web.shells[15].stress[i];

        printf("  epoch %d: loss=%.5f  inner(shell1) stress_sum=%.6f  outer(shell16) stress_sum=%.6f\n",
               epoch, loss, inner_stress_sum, outer_stress_sum);
    }

    
    long frozen_count = 0, total_links = 0, bad_weights = 0;
    for (int s = 0; s < 16 - 1; s++) {
        Weave *wv = &web.weaves[s];
        long n = (long)wv->rows * (long)wv->cols;
        total_links += n;
        for (long i = 0; i < n; i++) {
            if (wv->frozen_mask[i]) frozen_count++;
            float m = cabsf(wv->links[i]);
            if (isnan(m) || isinf(m)) bad_weights++;
        }
    }
    printf("\nPost-training: %ld/%ld links Crystallized (frozen), %ld weights are NaN/Inf.\n",
           frozen_count, total_links, bad_weights);
    printf(bad_weights == 0
        ? "No NaN/Inf weights after mutation. PASS.\n"
        : "FAILURE: corrupted weights detected.\n");

    web_free(&web);
}

int main(void) {
    printf("=== RadNet Skeleton Smoke Test ===\n");

    radnet_print_topology(16);
    test_silk();
    test_forward_pass();
    test_seed_and_full_pipeline();

    printf("\n-- Cylinder allocation test --\n");
    unsigned int rng_state = 12345u;
    Cylinder cyl;
    int num_webs = 4;

    if (cylinder_init(&cyl, num_webs, 16, 10, &rng_state) != 0) {
        fprintf(stderr, "Cylinder allocation FAILED\n");
        return 1;
    }
    printf("Allocated Cylinder with %d webs.\n", cyl.num_webs);

    
    Web *w0 = &cyl.webs[0];
    printf("Web0 Shell1 num_nodes=%d (expect 2)\n", w0->shells[0].num_nodes);
    printf("Web0 Shell16 num_nodes=%d (expect 181)\n", w0->shells[15].num_nodes);

    
    Weave *wv0 = &w0->weaves[0];
    printf("Web0 Weave(1->2) rows=%d cols=%d (expect rows=3 cols=2)\n",
           wv0->rows, wv0->cols);
    printf("Sample link[0]=%.4f%+.4fi frozen=%d\n",
           crealf(wv0->links[0]), cimagf(wv0->links[0]), wv0->frozen_mask[0]);

    cylinder_free(&cyl);
    printf("\nCylinder freed cleanly. Skeleton OK.\n");
    printf("\nNext steps (not yet implemented): FFTW3 seed, OpenBLAS cgemv forward pass,\n");
    printf("Echo stress backprop, mutation engine (Wind/Shift/Break/Crystallize).\n");
    return 0;
}

