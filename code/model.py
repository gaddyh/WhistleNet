from keras.layers import Dropout, Input, Dense, InputLayer, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Concatenate, Flatten, Reshape, Lambda, Embedding, dot, BatchNormalization
from keras.models import Model, load_model, Sequential
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC, BinaryAccuracy
import keras.backend as K
from keras.utils.vis_utils import plot_model
from tensorflow import keras
import os
import tensorflow as tf
from tensorflow.keras import regularizers

m = 20
n = 94
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc', from_logits=False),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def create_siamese_model() :
    input_layer = Input((m, n, 1))
    layer1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    layer2 = MaxPooling2D((2, 2), padding='same')(layer1)
    #layer25 = BatchNormalization()(layer2)
    layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(layer2)
    layer4 = MaxPooling2D((2, 2), padding='same')(layer3)
    layer5 = Flatten()(layer4)
    embeddings = Dense(30, activation=None)(layer5)
    #embeddings = Dense(3, activation=tf.keras.activations.exponential, kernel_regularizer=regularizers.l2(0.1))(layer5)

    norm_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    model = Model(inputs=input_layer, outputs=norm_embeddings)

    # Create siamese model
    input1 = Input((m, n, 1))
    input2 = Input((m, n, 1))

    # Create left and right twin models
    left_model = model(input1)
    right_model = model(input2)

    # Dot product layer
    dot_product = dot([left_model, right_model], axes=1, normalize=True)

    siamese_model = Model(inputs=[input1, input2], outputs=dot_product)

    # Model summary 
    print(siamese_model.summary())

    # Compile model    
    siamese_model.compile(optimizer='adam', loss= BinaryCrossentropy(from_logits=False), metrics=METRICS)
 
    # Plot flowchart fo model
    plot_model(siamese_model, to_file=os.getcwd()+'/siamese_model_mnist.png', show_shapes=1, show_layer_names=1)

    # Fit model
    # siamese_model.fit([X1, X2], y, epochs=100, batch_size=5, shuffle=True, verbose=True)

    # model.save(os.getcwd()+"/color_encoder.h5")
    # siamese_model.save(os.getcwd()+"/color_siamese_model.h5")

    return model, siamese_model
