
int mlp_debug(double *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%f\t", array[i]);
	}
	printf("\n");
	return 0;
}

void __mlp_rand(double *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (double)rand() / (double)RAND_MAX;
	}
}

int __mlp_max(int *arr, int size)
{
	int temp = 0;
	for (int i = 0; i < size; i++)
	{
		temp = (arr[i] > temp) ? arr[i] : temp;
	}
	return temp;
}

int __mlp_relu_func(double* x, int size)
{
	for(int i=0 ; i<size ; i++)
	{
		x[i] = (0 < x[i]) ? x[i] : 0;
	}
	return 0;
}

int __mlp_sigmoid_func(double* x, int size)
{
	for(int i=0 ; i<size ; i++)
	{
		x[i] = 1 / (1 + exp(-1 * x[i]));
	}
	return 0;
}

int __mlp_tanh_func(double* x, int size)
{
	for(int i=0 ; i<size ; i++)
	{
		x[i] = tanh(x[i]);
	}
	return 0;
}

int __mlp_softmax_func(double* x, int size)
{
	double total_exp = 0.0;
	for(int i=0 ; i<size ; i++)
	{
		x[i] = exp(x[i]);
		total_exp += x[i];
	}
	
	for(int i=0 ; i<size ; i++)
	{
		x[i] = x[i]/total_exp;
	}
	return 0;
}

int __mlp_multiply(
	double *input,
	double *weight,
	double *result,
	int in_size,
	int w_size,
	int r_size)
{
	// weight size should be equal to input size times result size
	if (w_size != (in_size * r_size))
		return 1;

	int weight_index = 0;
	for (int i = 0; i < r_size; i++)
	{
		for (int j = 0; j < in_size; j++)
		{
			result[i] += (input[j] * weight[weight_index]);
			weight_index++;
		}
	}
	return 0;
}

int __mlp_activate(
	double *arr,
	int arr_size,
	enum Mlp_Func func)
{
	int (*activation_func)(double*, int) = NULL;
	switch (func)
	{
	case MLP_RELU:
		activation_func = __mlp_relu_func;
		break;
	case MLP_TANH:
		activation_func = __mlp_tanh_func;
		break;
	case MLP_SIGMOID:
		activation_func = __mlp_sigmoid_func;
		break;
	case MLP_SOFTMAX:
		activation_func = __mlp_softmax_func;
		break;
	case MLP_LINEAR:
		return 0;
	default:
		return 1;
	}
	
	activation_func(arr, arr_size);

	return 0;
}

double mlp_norm(double value, double min, double max)
{
	return (value - min)/ (max-min);
}

double mlp_denorm(double ratio, double min, double max)
{
	return min + (max - min) * ratio;
}

struct Mlp_Network *mlp_create(				int input_size,
							   int hidden_size,
							   int hidden_multiplier,
							   int output_size,
							   enum Mlp_Func activation_func)
{
	srand(time(NULL));
	struct Mlp_Network *net = (struct Mlp_Network *)malloc(sizeof(struct Mlp_Network));
	//error checking
	if(!net) return NULL;

	net->input_size = input_size + 1; // +1 is bias
	net->output_size = output_size;
	net->hidden_size[0] = hidden_size + 1; //+1 is bias
	net->hidden_size[1] = hidden_multiplier;
	
	// memory allocation for buffer used to place the result of cross operation and activation
	int temp[3] = {net->input_size, net->hidden_size[0], net->output_size};
	net->buffer_size = __mlp_max(temp, 3);
	net->buffer = (double *)calloc(sizeof(double), net->buffer_size);
	net->temp = (double *)calloc(sizeof(double), net->buffer_size);
	//error checking
	if(!net->buffer || !net->temp)
	{
		mlp_destroy(net);
		return NULL;
	}
	
	// memory allocation for weights
	net->weight_size = (net->input_size * net->hidden_size[0]) +
					   (net->hidden_size[0] * net->hidden_size[0] * (net->hidden_size[1] - 1)) +
					   (net->hidden_size[0] * net->output_size);
	net->weights = (double *)calloc(sizeof(double), net->weight_size);
	//error checking
	if(!net->weights)
	{
		mlp_destroy(net);
		return NULL;
	}

	net->input_x_hidden = net->input_size * net->hidden_size[0];
	net->hidden_x_hidden = net->hidden_size[0] * net->hidden_size[0];
	net->hidden_x_output = net->hidden_size[0] * net->output_size;
	net->activation = activation_func;

	//randomize weights value
	__mlp_rand(net->weights, net->weight_size);
	return net;
}

int mlp_destroy(struct Mlp_Network *net)
{
	free(net->weights);
	free(net->buffer);
	free(net->temp);
	net->weights = NULL;
	net->buffer = NULL;
	net->temp = NULL;
	free(net);
	net = NULL;
	return 0;
}

int mlp_forward(struct Mlp_Network *net)
{
	int next_weights = 0;
	int input_x_hidden_size = 0;
	int hidden_x_hidden_size = 0;
	int hidden_x_output_size = 0;
	int loops = 0;
	
	// input layer X hidden layer operations
	net->buffer[net->input_size - 1] = 1.0; // set bias input value
	/*
	printf("\ninput\n");
	mlp_debug(net->buffer, net->input_size);
	printf("\nweights\n");
	mlp_debug(net->weights + next_weights, net->input_x_hidden);
	*/
	__mlp_multiply(net->buffer,
				   net->weights,
				   net->temp,
				   net->input_size,
				   net->input_x_hidden,
				   net->hidden_size[0]);
	memcpy(net->buffer, net->temp, sizeof(net->temp));
	memset(net->temp,0,sizeof(net->temp));
	__mlp_activate(net->buffer, net->hidden_size[0], MLP_RELU);
	/*
		printf("\nhidden_layer\n");
	mlp_debug(net->buffer, net->input_size);
	*/
	
	next_weights = net->input_x_hidden;

	// hidden_layer X hidden_layer operations
	loops = net->hidden_size[1] - 1;
	// loop through all of hidden layers
	for (int i = 0; i < loops; i++)
	{
		net->buffer[net->hidden_size[0] - 1] = 1.0; // set bias input value
		/*
		printf("\nweights\n");
		mlp_debug(net->weights + next_weights, net->hidden_x_hidden);
		*/
		__mlp_multiply(net->buffer,
					   net->weights + next_weights,
					   net->temp,
					   net->hidden_size[0],
					   net->hidden_x_hidden,
					   net->hidden_size[0]);
		memcpy(net->buffer, net->temp, sizeof(net->temp));
		memset(net->temp,0, sizeof(net->temp));
		__mlp_activate(net->buffer, net->hidden_size[0], MLP_RELU);
		/*
		printf("\nhidden_layer\n");
		mlp_debug(net->buffer, net->input_size);
		*/
		next_weights += net->hidden_x_hidden;
	}

	// hidden layer X output layer operation
	net->buffer[net->hidden_size[0] - 1] = 1.0; // set bias input value
	/*
	printf("\nweights\n");
	mlp_debug(net->weights + next_weights, net->hidden_x_output);
	*/
	__mlp_multiply(
		net->buffer,
		net->weights + next_weights,
		net->temp,
		net->hidden_size[0],
		net->hidden_x_output,
		net->output_size);
	memcpy(net->buffer, net->temp, sizeof(net->temp));
	memset(net->temp,0,sizeof(net->temp));
	/*
	printf("\noutput\n");
	mlp_debug(net->buffer, net->input_size);
	*/
	__mlp_activate(net->buffer, net->output_size, net->activation);
	/*
	printf("\ntanh\n");
	mlp_debug(net->buffer, net->input_size);
	*/
	memset(net->temp, 0, sizeof(net->temp));
	return 0;
}

/*
int mlp_training(struct Mlp_Network *net,
				 double lrate,
				 double target,
				 int max_loop,
				 struct Mlp_Dataset);
struct Mlp_Dataset *mlp_import_dataset(const char *file_name);
int mlp_export_dataset(struct Mlp_Dataset *dataset, const char *file_name);
*/

struct Mlp_Network *mlp_load(const char *filename)
{
	long file_size = 0;
	int buffer_size = 0;
	int weight_size = 0;
	int input_size = 0;
	int hidden_size[2] = {0};
	int output_size = 0;
	int activation = 0;
	char* str = NULL;
	char* str_weights = NULL;
	FILE* f = NULL;
	
	f = fopen(filename, "r");
	//error checking
	if(!f) return NULL;
	
	fseek(f,0,SEEK_END);
	file_size = ftell(f);
	//error checking
	if(file_size <= 0)
	{
		fclose(f);
		return NULL;
	}
	
	str = (char*)malloc(sizeof(char)*file_size);
	//error checking
	if(!str)
	{
		fclose(f);
		return NULL;
	}
	
	fseek(f, 0, SEEK_SET);
	fread(str,sizeof(char), file_size, f);
	fclose(f);
	//printf("%d: %s\n++++++++++++\n\n", file_size, str);
	
	str_weights = (char*)calloc(1, file_size);
	// error checking
	if(!str_weights)
	{
		free(str);
		str=NULL;
		return NULL;
	}
	
	sscanf(str, 
			"buffer_size=%d\n"
			"weight_size=%d\n"
			"input_size=%d\n"
			"hidden_size=%d %d\n"
			"output_size=%d\n"
			"activation=%d\n"
			"weights=%[^\n]", 
			&buffer_size, 
			&weight_size, 
			&input_size, 
			hidden_size,
			hidden_size+1,
			&output_size,
			&activation,
			str_weights);
	
	struct Mlp_Network* net = mlp_create(
	input_size-1, 
	hidden_size[0]-1, 
	hidden_size[1],
	output_size,
	(enum Mlp_Func)activation);
	//error checking
	if(!net)
	{
		free(str);
		free(str_weights);
		str = NULL;
		str_weights = NULL;
		return NULL;
	}
	
	/*
	printf(
			"buffer_size=%d\n"
			"weight_size=%d\n"
			"input_size=%d\n"
			"hidden_size=%d %d\n"
			"output_size=%d\n"
			"activation=%d\n", 
			net->buffer_size, 
			net->weight_size, 
			net->input_size, 
			net->hidden_size[0],
			net->hidden_size[1],
			net->output_size,
			net->activation);
	*/
	
	net->weights[0] = atof( strtok(str_weights, " ") );
	//printf("%f\n", net->weights[0]);
	for(int i=1 ; i < net->weight_size ; i++)
	{
		net->weights[i] = atof( strtok(NULL, " ") );
		//printf("%f\n", net->weights[i]);
	}
	
	free(str);
	free(str_weights);
	str=NULL;
	str_weights=NULL;
	return net;
}

int mlp_save(struct Mlp_Network *net, const char *filename)
{
	int str_size = 50 * 11 + net->weight_size;
	char *str = NULL;
	char f2str[50] = {0}; // f2str -> float/double to string
	FILE* f = NULL;
	
	str = (char *)calloc(sizeof(char), str_size);
	//error checking
	if(!str) return 1;	
	str_size = sprintf(str,
			"buffer_size=%d\n"
			"weight_size=%d\n"
			"input_size=%d\n"
			"hidden_size=%d %d\n"
			"output_size=%d\n"
			"activation=%d\n"
			"weights=",
			net->buffer_size, 
			net->weight_size, 
			net->input_size, 
			net->hidden_size[0], 
			net->hidden_size[1], 
			net->output_size, 
			net->activation);

	for (int i = 0; i < net->weight_size; i++)
	{
		str_size += sprintf(f2str, "%f ", net->weights[i]);
		strcat(str, f2str);
	}

	f = fopen(filename, "w");
	//error checking
	if(!f)
	{
		free(str);
		str = NULL;
		return 2;
	}
	fwrite(str, 1, str_size, f);
	fclose(f);
	return 0;
}

/*
int main(int argc, char *argv[])
{
	struct Mlp_Network *net = mlp_create(1, 1, 2, 1, MLP_TANH);
	//struct Mlp_Network* net = mlp_load("try_save_mlp.txt");
	
	printf("weight size = %d\n", net->weight_size);
	printf("buffer size = %d\n", net->buffer_size);
	printf("input size = %d\n", net->input_size);
	printf("hidden size = %d\n", net->hidden_size[0]);
	printf("hidden_multiplier = %d\n", net->hidden_size[1]);
	printf("output size = %d\n", net->output_size);

	net->buffer[0] = 0.2;

	net->weights[0] = 0.1;
	net->weights[1] = 0.2;
	net->weights[2] = 0.3;
	net->weights[3] = 0.4;
	net->weights[4] = 0.5;
	net->weights[5] = 0.6;
	net->weights[6] = 0.7;
	net->weights[7] = 0.8;
	net->weights[8] = 0.9;
	net->weights[9] = 1.0;
	
	mlp_forward(net);
	mlp_debug(net->buffer, net->buffer_size);
	//mlp_save(net, "try_save_mlp.txt");
	printf("coba normal: %f\n", mlp_denorm(0.5, -100.0, 100.0));
	mlp_destroy(net);
}
*/