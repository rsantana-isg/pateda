import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
import numpy as np


def get_optimizer(opt_method=0, learning_rate=1e-2):
    if opt_method==0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_method==1:
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif opt_method==2:        
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif opt_method==3:    
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_method==4:    
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    return optimizer

# Define the neural network D
def create_deblending_network(input_shape, output_shape):
    """
    Creates a simple neural network for deblending.
    
    Args:
    input_shape: The shape of the input tensor (x_alpha concatenated with alpha).
    
    Returns:
    A TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),  # Set input dtype to float64
        #tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(20, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(20, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64        
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)  # Output has the same dimension as x
    ])
    return model


def create_diff_network_0(input_shape, output_shape):
    """
    Creates a multilayer perceptron (MLP) based on the given architecture.
    
    Args:
    input_shape: The shape of the input tensor.
    
    Returns:
    A TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),  # Input layer
        
        # Layer 0
        tf.keras.layers.Dense(3, activation='softplus',
                              kernel_initializer=tf.keras.initializers.GlorotNormal(),
                              dtype=tf.float64),
        
        # Layer 1
        tf.keras.layers.Dense(7, activation='softplus',
                              kernel_initializer=tf.keras.initializers.RandomUniform(),
                              dtype=tf.float64),
        
        # Layer 2
        tf.keras.layers.Dense(4, activation='softsign',
                              kernel_initializer=tf.keras.initializers.RandomNormal(),
                              dtype=tf.float64),
        
        # Layer 3 (Output layer)
        tf.keras.layers.Dense(5, activation='softsign',
                              kernel_initializer=tf.keras.initializers.GlorotNormal(),
                              dtype=tf.float64),
        
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)  # Output has the same dimension as x
    ])
    
    return model


def create_diff_network(input_shape, output_shape):
    """
    Creates a multilayer perceptron (MLP) based on the given architecture.
    
    Args:
    input_shape: The shape of the input tensor.
    
    Returns:
    A TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),  # Input layer
        
        # Layer 0
        tf.keras.layers.Dense(6, activation='elu',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              dtype=tf.float64),
        
        # Layer 1 (No activation function, i.e., linear activation)
        tf.keras.layers.Dense(8, activation=None,
                              kernel_initializer=tf.keras.initializers.GlorotNormal(),
                              dtype=tf.float64),
        
        # Layer 2 (Output layer)
        tf.keras.layers.Dense(5, activation='tanh',
                              kernel_initializer=tf.keras.initializers.RandomNormal(),
                              dtype=tf.float64),

        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)  # Output has the same dimension as x
    ])
    
    return model




import tensorflow as tf

def nas_bbob_1_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(6*expand_factor, kernel_initializer='glorot_normal', activation=None),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_normal', activation='softsign'),
        tf.keras.layers.Dense(9*expand_factor, kernel_initializer='random_normal', activation='elu'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_normal', activation='softsign'),
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_4_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='glorot_normal', activation=None),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_uniform', activation=None),
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_6_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='random_uniform', activation='softplus'),
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_8_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_normal', activation='elu'),
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_9_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(6*expand_factor, kernel_initializer='glorot_normal', activation=None),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_normal', activation='softsign'),
        tf.keras.layers.Dense(9*expand_factor, kernel_initializer='random_normal', activation='elu'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_normal', activation='softsign'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_10_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='glorot_normal', activation=None),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_normal', activation='relu'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_normal', activation='elu'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

#def nas_bbob_11_10_15_20_20_2_111(input_shape, output_shape):
   

def nas_bbob_12_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(3*expand_factor, kernel_initializer='glorot_normal', activation='softsign'),
        tf.keras.layers.Dense(6*expand_factor, kernel_initializer='random_uniform', activation=None),
        tf.keras.layers.Dense(4*expand_factor, kernel_initializer='glorot_normal', activation='elu'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_13_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(4*expand_factor, kernel_initializer='random_normal', activation='elu'),
        tf.keras.layers.Dense(3*expand_factor, kernel_initializer='random_normal', activation=None),
        tf.keras.layers.Dense(3*expand_factor, kernel_initializer='glorot_normal', activation='elu'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_16_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='glorot_normal', activation=None),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='random_uniform', activation='softplus'),
        tf.keras.layers.Dense(3*expand_factor, kernel_initializer='glorot_uniform', activation=None),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

#def nas_bbob_17_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    

def nas_bbob_18_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_normal', activation='relu'),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='random_uniform', activation='sigmoid'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_19_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='random_uniform', activation='elu'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_uniform', activation='sigmoid'),
        tf.keras.layers.Dense(2*expand_factor, kernel_initializer='glorot_normal', activation='softsign'),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='random_uniform', activation='softplus'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_20_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='glorot_uniform', activation='elu'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='glorot_uniform', activation='softsign'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_normal', activation='elu'),
        tf.keras.layers.Dense(2*expand_factor, kernel_initializer='random_uniform', activation='elu'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_21_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='random_uniform', activation='softplus'),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='random_uniform', activation='relu'),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_uniform', activation=None),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_22_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(4*expand_factor, kernel_initializer='glorot_normal', activation='softsign'),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='random_normal', activation='relu'),
        tf.keras.layers.Dense(8*expand_factor, kernel_initializer='random_normal', activation='sigmoid'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_23_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(9*expand_factor, kernel_initializer='random_normal', activation='elu'),
        tf.keras.layers.Dense(9*expand_factor, kernel_initializer='glorot_normal', activation='sigmoid'),
        tf.keras.layers.Dense(6*expand_factor, kernel_initializer='random_normal', activation='tanh'),          
        tf.keras.layers.Dense(output_shape[0], activation='linear', dtype=tf.float64)
    ])
    return model

def nas_bbob_30_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='random_uniform', activation='softplus'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(7*expand_factor, kernel_initializer='random_uniform', activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5*expand_factor, kernel_initializer='glorot_uniform', activation=None),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(output_shape[0], activation='sigmoid', dtype=tf.float64)
    ])
    return model

def nas_bbob_31_10_15_20_20_2_111(input_shape, output_shape, expand_factor):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(9*expand_factor, kernel_initializer='random_normal', activation='elu'),
        tf.keras.layers.Dense(9*expand_factor, kernel_initializer='glorot_normal', activation='sigmoid'),
        tf.keras.layers.Dense(6*expand_factor, kernel_initializer='random_normal', activation='tanh'),          
        tf.keras.layers.Dense(output_shape[0], activation='sigmoid', dtype=tf.float64)
    ])
    return model


def nas_select_function(ind):
    nas_functions = [nas_bbob_1_10_15_20_20_2_111,
                     nas_bbob_4_10_15_20_20_2_111,
                     nas_bbob_6_10_15_20_20_2_111,
                     nas_bbob_8_10_15_20_20_2_111,
                     nas_bbob_9_10_15_20_20_2_111,
                     nas_bbob_10_10_15_20_20_2_111,
                     nas_bbob_12_10_15_20_20_2_111,
                     nas_bbob_13_10_15_20_20_2_111,
                     nas_bbob_16_10_15_20_20_2_111,
                     nas_bbob_18_10_15_20_20_2_111,
                     nas_bbob_19_10_15_20_20_2_111,
                     nas_bbob_20_10_15_20_20_2_111,
                     nas_bbob_21_10_15_20_20_2_111,
                     nas_bbob_22_10_15_20_20_2_111,
                     nas_bbob_23_10_15_20_20_2_111,
                     nas_bbob_30_10_15_20_20_2_111,
                     nas_bbob_31_10_15_20_20_2_111]
    return nas_functions[ind]
