import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from vit_keras import vit
from vit_utils import vit_b16
import resource
import gc
from subprocess import Popen, PIPE
from threading import Timer
import sys
import os
from mimic_tfds import load_caltech_birds2011

def port_pretrained_models(
    model_type='resnet50',
    input_shape=(224, 224, 3),
    num_classes=1000,
):
    if model_type == 'mobilenetv2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = True
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        
    elif model_type == 'resnet50':
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        base_model.trainable = True
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=True)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        print("Total layers in model:", len(base_model.layers))
        print("Trainable layers:", len([l for l in base_model.layers if l.trainable and l.trainable_weights]))

        
    elif model_type == 'vgg16':
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        base_model = tf.keras.applications.VGG16(input_shape=input_shape,
                                                 include_top=False,
                                                 weights='imagenet')
        base_model.trainable = True
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
    
    elif model_type == 'vit':
        # base_model = vit.vit_b16(
        #     image_size=input_shape[0],
        #     pretrained=True,
        #     include_top=True,
        #     pretrained_top=False,
        #     weights='imagenet21k+imagenet2012',
        #     classes=num_classes,
        # )
        base_model = vit_b16(
            image_size=input_shape[0],
            pretrained=True,
            include_top=True,
            pretrained_top=False,
            weights='imagenet21k+imagenet2012',
            classes=num_classes,
        )
        base_model.trainable = True
        # base_model.layers[4].layers[:-1]
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = vit.preprocess_inputs(x)
        outputs = base_model(x, training=False)
        model = tf.keras.Model(inputs, outputs)
    
    else:
        raise NotImplementedError("This model has not been implemented yet")
    
    return model



def port_datasets(
    dataset_name,
    input_shape,
    batch_size,
):
    """
    This function loads the train and test splits of the requested dataset, and
    creates input pipelines for training.

    Args:
        dataset_name (str): name of the dataset
        input_shape (tuple): NN input shape excluding batch dim
        batch_size (int): batch size of training split, 
        default batch size for testing split is batch_size*2
    
    Raises:
        NotImplementedError: The requested dataset is not implemented

    Returns:
        Train and test splits of the request dataset
    """
    
    # maximize number limit of opened files
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
    
    def prep(x, y):
        x = tf.image.resize(x, [input_shape[0], input_shape[1]])
        return x, y
                               
        
    # if dataset_name == 'caltech_birds2011':
    #     # ds = tfds.load('caltech_birds2011', as_supervised=True, download=False) # 200 classes
    #     ds = load_caltech_birds2011("dataset_caltech")
    #     ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
    #                              .batch(batch_size)\
    #                              .prefetch(buffer_size=tf.data.AUTOTUNE)
    #     ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
    #                            .batch(batch_size*2)\
    #                            .prefetch(buffer_size=tf.data.AUTOTUNE)
             
    if dataset_name == 'stanford_dogs':
       ds = tfds.load('stanford_dogs', as_supervised=True)
       ds['train'].shuffle(buffer_size=1000) # 120 classes
       ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                .batch(batch_size)\
                                .prefetch(buffer_size=tf.data.AUTOTUNE)
       ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                .batch(batch_size*2)\
                                .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    elif dataset_name == 'oxford_iiit_pet':
       ds = tfds.load('oxford_iiit_pet', as_supervised=True) # 37 classes
       ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                .batch(batch_size)\
                                .prefetch(buffer_size=tf.data.AUTOTUNE)
                                
       ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                              .batch(batch_size*2)\
                              .prefetch(buffer_size=tf.data.AUTOTUNE)

    elif dataset_name == 'flowers102':
       ds = tfds.load('oxford_flowers102', as_supervised=True) # 37 classes
       ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                .batch(batch_size)\
                                .prefetch(buffer_size=tf.data.AUTOTUNE)
                                
       ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                              .batch(batch_size*2)\
                              .prefetch(buffer_size=tf.data.AUTOTUNE)
                              
    else:
        raise NotImplementedError("This dataset has not been implemented yet")
                              
    return ds_train, ds_test
