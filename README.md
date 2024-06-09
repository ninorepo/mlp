# mlp
Simple to use Multilayer Perceptron Library

# usage
## Creating & Destroying MLP
```c
int input_size = 1;
int hidden_size = 1;
int hidden_layers = 2;
int output_size = 1;

struct Mlp_Network *net = mlp_create(input_size, hidden_size, hidden_layers, output_size, MLP_TANH);
mlp_destroy(net);
```

## Run Feedforward

```c
double output = 0;

struct Mlp_Network *net = mlp_create(1, 1, 2, 1, MLP_TANH);
net->buffer[0] = 0.2; // set input
mlp_forward(net);
output = net->buffer[0]; // get output
mlp_destroy(net);
```

## Accessing weights
```c
double weights = net->weights;
int length = net->weight_size;
```

## Save to text file
```c
mlp_save(net, "mlp_file");
```

## Load file into MLP object

```c
struct Mlp_Network* net = mlp_load("mlp_file");
```

