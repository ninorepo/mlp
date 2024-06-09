#include "stdio.h"
#include "stdlib.h"
#include "math.h" // exp()
#include "string.h"
#include "time.h"

enum Mlp_Func
{
	MLP_RELU,
	MLP_TANH,
	MLP_LINEAR,
	MLP_SIGMOID,
	MLP_SOFTMAX
};

/*
struct Mlp_Dataset
{
	double *features; 
	double *labels;	  
	int feature_size;
	int label_size;
};
*/

struct Mlp_Network
{
	double *temp;
	double *buffer;
	double *weights;
	int buffer_size;
	int weight_size;
	int input_size;
	int output_size;
	int hidden_size[2];
	int input_x_hidden;
	int hidden_x_hidden;
	int hidden_x_output;
	enum Mlp_Func activation;
};

int mlp_debug(double *array, int size);
void __mlp_rand(double *array, int size);
int __mlp_max(int *arr, int size);
int __mlp_relu_func(double* x, int size);
int __mlp_sigmoid_func(double* x, int size);
int __mlp_tanh_func(double* x, int size);
int __mlp_softmax_func(double* x, int size);
int __mlp_multiply(double *input, double *weight, double *result, int in_size, int w_size, int r_size);
int __mlp_activate(double *arr, int arr_size, enum Mlp_Func func);
double mlp_normal(double value, double min, double max);
double mlp_denormal(double ratio, double min, double max);
struct Mlp_Network *mlp_create(int input_size, int hidden_size, int hidden_multiplier, int output_size, enum Mlp_Func activation_func);
int mlp_destroy(struct Mlp_Network *net);
int mlp_forward(struct Mlp_Network *net);
struct Mlp_Network *mlp_load(const char *filename);
int mlp_save(struct Mlp_Network *net, const char *filename);


