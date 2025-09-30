import tensorflow as tf
from tensorflow.keras import layers, models, activations
import numpy as np

from tensorflow.keras.models import Model

def profile_memory_cost(net, trainable_params, input_size=(1, 3, 224, 224), require_backward=True,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=4, batch_size=8):
	param_size = count_model_size(net, trainable_params, trainable_param_bits, frozen_param_bits, print_log=True)
	activation_size, _ = count_activation_size(net, trainable_params, input_size, require_backward, activation_bits)

	memory_cost = activation_size * batch_size + param_size
	return memory_cost, {'param_size': param_size, 'act_size': activation_size}


def count_model_size(model, trainable_params, trainable_param_bits=32, frozen_param_bits=4, print_log=True):
    """
    Calculate the size of a Keras model in bytes based on trainable and non-trainable parameters.
    
    Args:
        model: A Keras model object.
        trainable_param_bits: Bits per parameter for trainable variables (default: 32).
        frozen_param_bits: Bits per parameter for frozen variables (default: 8).
        print_log: Whether to print a log of the sizes (default: True).
    
    Returns:
        Total model size in bytes.
    """
    frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

    trainable_param_size = 0
    frozen_param_size = 0
    # print(len(trainable_params))
    # Iterate over all weights in the model
    for weight in model.weights:
        # print(weight)
        param_count = tf.size(weight).numpy()  # Get the total number of parameters
        # if weight in model.trainable_weights:
        if any(np.array_equal(weight.numpy(), w.numpy()) for w in trainable_params):
            trainable_param_size += trainable_param_bits / 8 * param_count
        else:
            frozen_param_size += frozen_param_bits / 8 * param_count

    # Calculate total model size
    model_size = trainable_param_size + frozen_param_size

    # Optionally print the log
    if print_log:
        print(f'Total: {model_size:.0f} bytes',
              f'\tTrainable: {trainable_param_size:.0f} bytes (data bits {trainable_param_bits})',
              f'\tFrozen: {frozen_param_size:.0f} bytes (data bits {frozen_param_bits})')
    
    return model_size


def count_activation_size(model, trainable_params, input_size=(1, 224, 224, 3), require_backward=True, activation_bits=8, print_log=True):
    act_byte = tf.cast(activation_bits / 8, tf.int32)
    memory_info_dict = {
        'peak_activation_size': tf.Variable(0, dtype=tf.float32),
        'grad_activation_size': tf.Variable(0, dtype=tf.float32),
        'residual_size': tf.Variable(0, dtype=tf.float32),
    }
    # print(model.summary)
    
    def extract_unique_layer_names(trainable_params):
        names = []
        for w in trainable_params:
            # print(w.name)
            names += [w.name.split('/')[0]]
        # Convert the set back to a list and return
        return list(names)
    
    trainable_layers = extract_unique_layer_names(trainable_params)
    # print(trainable_layers)

    def count_convNd(layer, trainable_layers, x, y):
        if layer.name in trainable_layers and layer.weights is not None:
            num_elements = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
            layer.grad_activations = tf.cast(num_elements, tf.int32) * act_byte  # Size in bytes
        else:
            layer.grad_activations = 0

        # Temporary memory footprint required by inference (tmp_activations)
        inp_size = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
        out_size = tf.reduce_prod(tf.shape(y))  # Number of elements in output tensor
        layer.tmp_activations = inp_size * act_byte + (out_size *  act_byte // layer.groups)  # Size in bytes

    def count_linear(layer, trainable_layers, x, y):
        if layer.name in trainable_layers and layer.weights is not None:
            num_elements = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
            layer.grad_activations = tf.cast(num_elements, tf.int32) * act_byte  # Size in bytes
        else:
            layer.grad_activations = 0

        # Temporary memory footprint required by inference (tmp_activations)
        inp_size = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
        out_size = tf.reduce_prod(tf.shape(y))  # Number of elements in output tensor
        layer.tmp_activations = inp_size * act_byte + (out_size * act_byte)

    def count_bn(layer, trainable_layers, x, _):
        if layer.name in trainable_layers and layer.weights is not None:
            num_elements = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
            layer.grad_activations = tf.cast(num_elements, tf.int32) * act_byte  # Size in bytes
        else:
            layer.grad_activations = 0

        # Temporary memory footprint required by inference (tmp_activations)
        inp_size = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
        layer.tmp_activations = inp_size * act_byte


    def count_relu(layer, trainable_layers, input, output):
        if layer.name in trainable_layers and require_backward:
            num_elements = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
            layer.grad_activations = tf.cast(num_elements/8, tf.int32)  # Size in bytes
        else:
            layer.grad_activations = 0

        # Temporary memory footprint required by inference (tmp_activations)
        inp_size = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
        layer.tmp_activations = inp_size * act_byte

    def count_smooth_act(layer, trainable_layers, input, output):
        if layer.name in trainable_layers and require_backward:
            num_elements = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
            layer.grad_activations = tf.cast(num_elements, tf.int32) * act_byte  # Size in bytes
        else:
            layer.grad_activations = 0

        # Temporary memory footprint required by inference (tmp_activations)
        inp_size = tf.reduce_prod(tf.shape(x[0]))  # Number of elements in input tensor
        layer.tmp_activations = inp_size * act_byte


    def get_activations(model, input_data):
        """
        This function recursively extracts activations from each layer of the model
        using the Keras API.
        """
        activations = {}

        # Loop over all layers of the model
        for layer in model.layers:
            # Handle sub-models like Sequential or Functional models (which have their own layers)
            if isinstance(layer, tf.keras.Model):
                sub_activations = get_activations(layer, input_data)
                activations.update(sub_activations)
            elif hasattr(layer, "output"):
                # print(layer.name)
                # try:
                    # If the layer has an output, create an intermediate model to capture activations
                intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                output = intermediate_model(input_data)
                if str(type(output)) == "<class 'tuple'>": # Check if it's a tuple 
                    output = output[0]  
                activations[layer.name] = output.numpy()
                if len(list(activations.values()))>1:
                    x = list(activations.values())[-2]
                else:
                    x = input_data
                
                y = list(activations.values())[-1]

                if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Conv3D) or isinstance(layer, layers.Conv1D):
                    count_convNd(layer, trainable_layers, x, y)
                elif isinstance(layer, layers.Dense):
                    count_linear(layer, trainable_layers, x, y)
                elif isinstance(layer, layers.BatchNormalization):
                    count_bn(layer, trainable_layers, x, y)
                elif isinstance(layer, layers.ReLU):
                    count_relu(layer, trainable_layers, x, y)
                # elif isinstance(layer, layers.Sigmoid) or isinstance(layer,layers.Tanh):
                #     count_smooth_act(layer, trainable_layers,x, y)
                else:
                    layer.forward_hook = None

                if hasattr(layer, 'tmp_activations'):
                    current_act_size = layer.tmp_activations.numpy() + memory_info_dict['grad_activation_size'] 
                    layer.act_size = current_act_size
                    memory_info_dict['peak_activation_size'] = max(current_act_size, memory_info_dict['peak_activation_size'])
					# memory_info_dict['grad_activation_size'] += _module.grad_activations

                    # peak_size = max(memory_info_dict['peak_activation_size'].numpy(), layer.tmp_activations.numpy())
                    # memory_info_dict['peak_activation_size'].assign(peak_size)
                    memory_info_dict['grad_activation_size'].assign_add(tf.cast(layer.grad_activations, tf.float32))


        return activations

    x = np.zeros(input_size, dtype=np.float32)
    act = get_activations(model, x)

    if print_log:
        print('Peak Activation Size:' , memory_info_dict['peak_activation_size'].numpy())
        print('Grad Activation Size:' , memory_info_dict['grad_activation_size'].numpy())

    return memory_info_dict['peak_activation_size'].numpy(), memory_info_dict['grad_activation_size'].numpy()

