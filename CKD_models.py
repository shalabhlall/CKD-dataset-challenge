import tensorflow as tf
from CKD_utils import *
from keras import regularizers

def GRU_model():
    model_0 = tf.keras.models.Sequential()
    model_0.add(tf.keras.layers.GRU(128, input_shape = (700, 27), activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model_0.add(tf.keras.layers.Dropout(0.1))
    model_0.add(tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model_0.add(tf.keras.layers.Dropout(0.1))
    model_0.add(tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model_0.add(tf.keras.layers.Dropout(0.1))
    model_0.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    
    model_0.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', f1_factor])
    
    return model_0
    
    
    
def CNN_GRU_model():
    main_input = tf.keras.layers.Input(shape=(700, 27), name='main_input')
    z1 = tf.keras.layers.Conv1D(128, 5, activation="relu")(main_input)
    z1 = tf.keras.layers.MaxPooling1D(pool_size=3)(z1)
    z1 = tf.keras.layers.Dropout(0.2)(z1)
    z1 = tf.keras.layers.Conv1D(64, 5,  activation="relu")(z1)
    z1 = tf.keras.layers.MaxPooling1D(pool_size=2)(z1)
    z1 = tf.keras.layers.Dropout(0.2)(z1)
    z1 = tf.keras.layers.Conv1D(64, 5,  activation="relu")(z1)
    z1 = tf.keras.layers.MaxPooling1D(pool_size=2)(z1)
    z1 = tf.keras.layers.Dropout(0.2)(z1)
    z1 = tf.keras.layers.GRU(64, activation="relu", return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(z1)
    z1 = tf.keras.layers.Dropout(0.1)(z1)
    gru_out = tf.keras.layers.GRU(64, activation="relu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(z1)

    auxiliary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(gru_out)
    auxiliary_input = tf.keras.layers.Input(shape=(7,), name='aux_input')

    z2 = tf.keras.layers.Concatenate(axis=1)([gru_out, auxiliary_input])

    z2 = tf.keras.layers.Dropout(0.2)(z2)
    z2 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(z2)
    z2 = tf.keras.layers.Dropout(0.2)(z2)
    z2 = tf.keras.layers.Dense(16, activation="relu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(z2)
    z2 = tf.keras.layers.Dropout(0.2)(z2)
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')(z2)

    model = tf.keras.models.Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', f1_factor])
    
    return model 



def inceptionTime_model():
    input_layer = tf.keras.layers.Input(shape=(700, 27), name='main_input')
    nb_classes = 3

    x = input_layer
    input_res = input_layer

    for d in range(nb_classes):

        x = inceptionModule(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        if d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(gap_layer)

    auxiliary_input = tf.keras.layers.Input(shape=(7,), name='aux_input')

    z2 = tf.keras.layers.Concatenate(axis=1)([gap_layer, auxiliary_input])

    z2 = tf.keras.layers.Dropout(0.2)(z2)
    z2 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(z2)
    z2 = tf.keras.layers.Dropout(0.2)(z2)
    z2 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(z2)
    z2 = tf.keras.layers.Dropout(0.2)(z2)
    main_output = tf.keras.layers.Dense(1, activation='sigmoid')(z2)

    model_net = tf.keras.models.Model(inputs=[input_layer, auxiliary_input], outputs=[main_output])
    
    model_net.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', f1_factor])

    
    return model_net



def inceptionModule(inputs):
    Z_bottleneck = tf.keras.layers.Conv1D(filters = 32, kernel_size = 1, activation="relu")(inputs)
    Z_maxpool = tf.keras.layers.MaxPooling1D(pool_size=3, strides = 1, padding = 'same')(inputs)
    
    Z1 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 10, padding='same', activation="relu")(Z_bottleneck)
    Z2 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 20, padding='same', activation="relu")(Z_bottleneck)
    Z3 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 40, padding='same', activation="relu")(Z_bottleneck)
    Z4 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 1, padding='same', activation="relu")(Z_maxpool)
    
    Z = tf.keras.layers.Concatenate(axis=2)([Z1, Z2, Z3, Z4])
    Z = tf.keras.layers.BatchNormalization()(Z)
    Z = tf.keras.layers.Activation(activation='relu')(Z)
    return Z



def shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x
    
