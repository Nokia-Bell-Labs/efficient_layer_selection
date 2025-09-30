# © 2025 Nokia
# Licensed under the BSD 3-Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import tensorflow as tf
from ripser import ripser
import numpy as np

@tf.function
def train_step(x, y, model, loss_fn_cls, var_list, optimizer, accuracy, cls_loss):
        with tf.device('/GPU:7'):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn_cls(y, y_pred)
            gradients = tape.gradient(loss, var_list)
            optimizer.apply_gradients(zip(gradients, var_list))
            accuracy(y, y_pred)
            cls_loss(loss)

@tf.function
def test_step(x, y, model, loss_fn_cls, accuracy, cls_loss):
    with tf.device('/GPU:7'):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)


def get_activations_and_gradients(model, input_data, output_data=None, loss_fn=None, requires_gradients=False):
    """
    Returns activations and gradients w.r.t. activations for layers with trainable weights in a Keras model.

    Args:
        model: tf.keras.Model
        input_data: input tensor (e.g., a batch of images)
        output_data: ground truth tensor
        loss_fn: loss function (e.g., tf.keras.losses.CategoricalCrossentropy())

    Returns:
        activations: dict mapping layer names to activation tensors
        gradients: dict mapping layer names to gradient tensors
    """
    activations = {}
    gradients = {}

    def collect_activations(layer, x, tape, parent_name=""):
        # Compose full layer name
        layer_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
        # print(layer_name)
        # Recurse if it's a model
        if isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                x = collect_activations(sublayer, x, tape, parent_name=layer_name)
            return x

        # If it's a transformer block, manually collect only *direct* sublayers
        elif "encoderblock" in layer.name and "Dense" not in layer.name:
            visited = set()
            for sublayer in layer.submodules:
                # print(sublayer.name)
                if sublayer == layer or sublayer.name in visited:
                    continue
                if any(skip in sublayer.name for skip in ["query", "key", "value", "out"]):
                    continue
                visited.add(sublayer.name)
                x = collect_activations(sublayer, x, tape, parent_name=layer_name)
            return x

        # Leaf layers
        else:
            try:
                # if layer.trainable and layer.trainable_weights:
                if layer.trainable_weights and hasattr(layer, '__call__'):
                    x = layer(x)
                    tape.watch(x)
                    if isinstance(x, tuple):
                        x = x[0]
                    activations[layer_name] = x
                else:
                    x = layer(x)
            except Exception as e:
                print(f"Skipping {layer_name}")
            return x

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_data)
        x = collect_activations(model, input_data, tape)
        if requires_gradients == True:
            loss = loss_fn(output_data, x)
    
    if requires_gradients == True:
        for layer_name, activation in activations.items():
            grad = tape.gradient(loss, activation)
            if grad is not None:
                gradients[layer_name] = grad.numpy()
            else:
                gradients[layer_name] = None  # Can help for debugging missing grads

    return activations, gradients



def compute_betti_activations(model, x):
    act, g = get_activations_and_gradients(model, x)
    betti = compute_normalized_betti_numbers(act)
    w_dict = betti
    return w_dict

def compute_normalized_betti_numbers(act):
    betti = {}
    for name in act.keys():
        if isinstance(act[name], tuple):
            act[name] = act[name][0]
        activation = np.array(act[name])
        a = activation.reshape((activation.shape[0],-1)) #batch_size
        result = ripser(a)
        # Step 5: Plot the persistence diagram
        diagrams = result['dgms']
        # plot_diagrams(diagrams, show=True)
        # Step 6: Extract Betti numbers from the persistence diagram
        betti_numbers = [sum(1 for interval in dgm if interval[1] > interval[0]) for dgm in diagrams]
        # print(betti_numbers)
        betti[name] = betti_numbers[1]/tf.reduce_prod(tf.shape(a))

    return betti

def sort_keys(I_dict, rho):
    keys = list(I_dict.keys())
    vals = list(I_dict.values()) 
    sorted_keys = [key for key, _ in sorted(zip(keys, vals), key=lambda x: x[1], reverse=True)]
    print(f"choosing {int(rho * len(sorted_keys))} layers out of {len(sorted_keys)} layers")
    sorted_keys = sorted_keys[:int(rho*len(sorted_keys))]
    m = []
    for key in I_dict:
        if key in sorted_keys:
            m.append(1)
        else:
            m.append(0)
    # print(m)
    return I_dict, m

def final_prints(best_validation_acc, total_time_0, epochs, sel_time, memory_cost, rho):
    best_validation_acc = best_validation_acc.numpy() * 100
    total_time_0 /= 3600
    print('===============================================')
    print('Training Type: Act Normalized Betti Train')
    print(f"Accuracy (%): {best_validation_acc:.2f}")
    print(f"Time (h): {total_time_0:.2f}")
    print(f"Time per epoch (s): {(total_time_0/epochs)*3600:.2f}")
    print(f"Selection Time (s): {(sel_time):.2f}")
    print(f"Memory (MB): {(memory_cost*1e-6):.2f}")
    print(f"rho: {rho:.2f}")
    print('===============================================')


def get_fisher(a_dict, gradients):
    fisher = {}
    for num, layer in enumerate(a_dict):
        if gradients[layer] is not None:
            g_nk = tf.multiply(a_dict[layer], gradients[layer])
            fisher[layer] = tf.reduce_mean(tf.pow(g_nk,2))/2
        else:
            fisher[layer] = 0

    return fisher

