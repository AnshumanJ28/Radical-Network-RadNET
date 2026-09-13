#ifndef RADNET_H
#define RADNET_H
#ifdef __cplusplus
#include <complex>
typedef std::complex<float> radnet_complex;
extern "C" {
#else
#include <complex.h>
typedef float complex radnet_complex;
#endif

#include <stdbool.h>



typedef struct {
    int num_nodes;
    radnet_complex *z_state;
    radnet_complex *seed;
    radnet_complex *err_vec;
    float *gates;
    float *stress;
} Shell;


typedef struct {
    int rows;   
    int cols;   
    radnet_complex *links;
    radnet_complex *bias;
    bool *frozen_mask;
    int *stable_epochs;
} Weave;


typedef struct {
    int n;
    radnet_complex *weights; 
    radnet_complex *bias;    
    bool *frozen_mask;
    int *stable_epochs;
} ForgetGate;


typedef struct {
    int num_shells;
    Shell *shells;
    Weave *weaves;             
    ForgetGate *forget_gates;  
} Web;


typedef struct {
    int input_size;
    int output_size;
    radnet_complex *weights;
    float *bias;
    bool *frozen_mask;
    int *stable_epochs;
    radnet_complex *err_vec;
    radnet_complex *z_in;
    radnet_complex *z_out;
} TradeoffLayer;


typedef struct {
    int num_webs;
    Web *webs;
    TradeoffLayer tradeoff;
} Cylinder;


int  radnet_shell_node_count(int shell_index_1based);
void radnet_print_topology(int num_shells);


int  shell_init(Shell *sh, int num_nodes);
void shell_free(Shell *sh);

int  weave_init(Weave *w, int rows, int cols, unsigned int *rng_state);
void weave_free(Weave *w);

int  forget_gate_init(ForgetGate *fg, int n, unsigned int *rng_state);
void forget_gate_free(ForgetGate *fg);

int  web_init(Web *web, int num_shells, unsigned int *rng_state);
void web_free(Web *web);

int  tradeoff_layer_init(TradeoffLayer *tl, int in_size, int out_size, unsigned int *rng_state);
void tradeoff_layer_free(TradeoffLayer *tl);

int  cylinder_init(Cylinder *cyl, int num_webs, int num_shells, int out_classes, unsigned int *rng_state);
void cylinder_free(Cylinder *cyl);


radnet_complex silk_activate(radnet_complex z, float gate_bias);
void  silk_activate_vec(const radnet_complex *in, radnet_complex *out,
                         int n, float gate_bias);


int radnet_forward_shell(Web *web, int s, float silk_gate_bias);


int radnet_seed_from_image(const float *pixels, int width, int height,
                            Web *web);


int radnet_compute_snag(const Web *web, int label, float *loss_out);


void radnet_compute_echo(Web *web, int label, float lambda);


int cylinder_forward(Cylinder *cyl, float *out_probs);
void cylinder_echo(Cylinder *cyl, const float *out_probs, int label, float lambda);
void cylinder_mutate(Cylinder *cyl, float thresh_shift, float thresh_break,
                     float low_epsilon, float learning_rate, unsigned int *rng_state);


void radnet_cgemv(int rows, int cols, const radnet_complex* A, const radnet_complex* x, radnet_complex* y);
void radnet_cgemv_conj(int rows, int cols, const radnet_complex* A, const radnet_complex* x, radnet_complex* y);


void radnet_mutate_web(Web *web, float thresh_shift, float thresh_break,
                       float low_epsilon, float learning_rate, unsigned int *rng_state);

#ifdef __cplusplus
}
#endif

#endif 

