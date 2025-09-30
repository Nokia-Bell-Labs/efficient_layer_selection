# © 2025 Nokia
# Licensed under the BSD 3-Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import time
from tqdm import tqdm
import os
from utils import clear_cache_and_rec_usage
from ripser import ripser
from memory_utils import profile_memory_cost
from train_utils import train_step, test_step, compute_betti_activations, sort_keys, final_prints, get_activations_and_gradients, get_fisher
from collections import defaultdict
import re
from utils import port_pretrained_models

def betti_training(
    model,
    model_name,
    ds_train,
    ds_test,
    run_name,
    logdir,
    timing_info,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
    interval=4,
    rho=0.1,
    disable_random_id=False,
    save_model=False,
    save_txt=False,
):
    """Train with Betti Trainer"""

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs
        
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    if disable_random_id:
        runid = run_name
    else:
        runid = run_name + '_elastic_x' + str(np.random.randint(10000))
    
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []

    training_step = 0
    best_validation_acc = 0
    
    total_time_0 = 0
    total_time_1 = 0
    
    t_sel_0 = time.time()
    print("## Parameter selection using Betti")
    # if epoch % interval == 0:
    batch_image, batch_label = next(iter(ds_train))

    num_batches_for_betti = 4

    I = defaultdict(lambda: 0)
    for x, y in tqdm(ds_train.take(num_batches_for_betti)):
        # print(i)
        _I_dict = compute_betti_activations(model, x)
        for k, v in _I_dict.items():
            I[k] += v
            _I = {k: v / len(I.keys()) for k, v in I.items()}

    _, m = sort_keys(_I, rho) #2
    # print(I.keys())
    print(f"num_batches_for_betti {num_batches_for_betti}")
    print(m) 

    if model_name == "vit":
        I_stripped = dict.fromkeys([key.replace("model/", "") for key in I])
        I_stripped = dict.fromkeys([re.sub(r"model_\d+/", "", key) for key in I_stripped])
        I_stripped = dict.fromkeys([key.replace("vit-b16/", "") for key in I_stripped])
        I_stripped = dict.fromkeys(
        [re.sub(r"/MlpBlock_\d+/Transformer/encoderblock_\d+", "", key) for key in I_stripped])
        # print(I_stripped.keys())

    else:
        I_stripped = dict.fromkeys([key.replace("model/", "") for key in I])
        I_stripped = dict.fromkeys([re.sub(r"model_\d+/", "", key) for key in I_stripped])
        if model_name == "resnet50":
            I_stripped = dict.fromkeys([key.replace("resnet50/", "") for key in I_stripped])
        if model_name == "mobilenetv2":
            I_stripped = dict.fromkeys([key.replace("mobilenetv2_1.00_224/", "") for key in I_stripped])

    var_list = []
    all_vars = model.trainable_weights
    # for var in all_vars:
    #     print(var.name)

    layer_flags = dict(zip(list(I_stripped.keys()), m))

    if model_name == "vit":
        var_list = [
            var
            for var in all_vars
            if layer_flags.get(var.name.rsplit('/', 1)[0],0) == 1
        ]
    else:
        var_list = [
            var
            for var in all_vars
            if layer_flags.get(var.name.split('/')[0], 0) == 1
        ]
    # print(len(var_list)) 
    # for var in var_list:
    #     print(var.name)
    # print(len(var_list)) 
    t_sel_1 = time.time()
    sel_time = (t_sel_1 - t_sel_0)
    print(sel_time)
    
    inp_size = batch_image.shape
    memory_cost, size_dict = profile_memory_cost(model, var_list, input_size=inp_size, require_backward=False)
    print("Memory consumption: ", memory_cost*1e-6)
    
    
        

    train_step_cpl = tf.function(train_step)

        
    for epoch in range(epochs):   
        t0 = time.time()            
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            train_step(x, y, model, loss_fn_cls, var_list, optimizer, accuracy, cls_loss)


        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        # print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y, model, loss_fn_cls, accuracy, cls_loss)


        if accuracy.result() > best_validation_acc:
            best_validation_acc = accuracy.result()
        print("=================================")
        print("acc: ", accuracy.result()* 100)
        print("loss: ", cls_loss.result())
        print("=================================")
        cls_loss.reset_states()
        accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)
    
    # print("total time excluding validation (s):", total_time_0)
    # print("total time including validation (s):", total_time_1)
    final_prints(best_validation_acc, total_time_0, epochs, sel_time, memory_cost, rho)
    if save_txt:
        np.savetxt(logdir + '/' + runid + '.txt', np.array([total_time_0, best_validation_acc]))
    # sig_stop_handler(None, None)
    del model
    del cls_loss
    del accuracy
    del optimizer


def fisher_training(
    model,
    model_name,
    ds_train,
    ds_test,
    run_name,
    logdir,
    timing_info,
    optim='sgd',
    lr=1e-4,
    weight_decay=5e-4,
    epochs=12,
    interval=4,
    rho=0.533,
    disable_random_id=False,
    save_model=False,
    save_txt=False,
):
    """Train with Betti Trainer"""

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs  
        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    
    # loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    if disable_random_id:
        runid = run_name
    else:
        runid = run_name + '_elastic_x' + str(np.random.randint(10000))
    
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    # accuracy = tf.metrics.SparseCategoricalAccuracy()
    # cls_loss = tf.metrics.Mean()

    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []

    training_step = 0
    best_validation_acc = 0
    
    total_time_0 = 0
    total_time_1 = 0
    
    t_sel_0 = time.time()
    print("## Parameter selection using Fisher Information")
    # if epoch % interval == 0:
    batch_image, batch_label = next(iter(ds_train))


    activations_o, gradients_o = get_activations_and_gradients(model, batch_image, output_data=batch_label, loss_fn=loss_fn_cls, requires_gradients=True)
    # activations = {k: v for k, v in activations_o.items()}
    # gradients = {k: v for k, v in gradients_o.items()}
    # print(activations_o.keys())
    # print(gradients_o.keys())
    fisher = get_fisher(activations_o, gradients_o)


    I, m = sort_keys(fisher, rho) #2

    # I = compute_betti_activations(model, batch_image) #2
    # I, m = sort_keys(I, rho) #2
    print(m)

    if model_name == "vit":
        I_stripped = dict.fromkeys([key.replace("model/", "") for key in I])
        I_stripped = dict.fromkeys([re.sub(r"model_\d+/", "", key) for key in I_stripped])
        I_stripped = dict.fromkeys([key.replace("vit-b16/", "") for key in I_stripped])
        I_stripped = dict.fromkeys(
        [re.sub(r"/MlpBlock_\d+/Transformer/encoderblock_\d+", "", key) for key in I_stripped])
        # print(I_stripped.keys())

    else:
        I_stripped = dict.fromkeys([key.replace("model/", "") for key in I])
        I_stripped = dict.fromkeys([re.sub(r"model_\d+/", "", key) for key in I_stripped])
        if model_name == "resnet50":
            I_stripped = dict.fromkeys([key.replace("resnet50/", "") for key in I_stripped])
        if model_name == "mobilenetv2":
            I_stripped = dict.fromkeys([key.replace("mobilenetv2_1.00_224/", "") for key in I_stripped])

    var_list = []
    all_vars = model.trainable_weights
    # for var in all_vars:
    #     print(var.name)

    layer_flags = dict(zip(list(I_stripped.keys()), m))

    if model_name == "vit":
        var_list = [
            var
            for var in all_vars
            if layer_flags.get(var.name.rsplit('/', 1)[0],0) == 1
        ]
    else:
        var_list = [
            var
            for var in all_vars
            if layer_flags.get(var.name.split('/')[0], 0) == 1
        ]
    print(len(var_list)) 
    t_sel_1 = time.time()
    sel_time = (t_sel_1 - t_sel_0)
    print(sel_time)

    train_step_cpl = tf.function(train_step)
    inp_size = batch_image.shape
    memory_cost, size_dict = profile_memory_cost(model, model.trainable_weights, input_size=inp_size)
    print("Memory consumption: ", memory_cost*1e-6)
    
    

    
    for epoch in range(epochs):   
        t0 = time.time()            
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1

            train_step(x, y, model, loss_fn_cls, var_list, optimizer, accuracy, cls_loss)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()


        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y, model, loss_fn_cls, accuracy, cls_loss)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                if save_model:
                    model.save_weights(os.path.join('saved_models', runid + '.tf'))
            print("=================================")
            print("acc: ", accuracy.result()* 100)
            print("loss: ", cls_loss.result())
            print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)
    
    # print("total time excluding validation (s):", total_time_0)
    # print("total time including validation (s):", total_time_1)
    final_prints(best_validation_acc, total_time_0, epochs, sel_time, memory_cost, rho)
    if save_txt:
        np.savetxt(logdir + '/' + runid + '.txt', np.array([total_time_0, best_validation_acc]))
    # sig_stop_handler(None, None)
