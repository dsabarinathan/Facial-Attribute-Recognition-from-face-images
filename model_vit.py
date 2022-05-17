# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:38:13 2022

@author: SABARI
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def vitNet():
    def __init__(self,image_size=128,batch_size=4,num_epochs=100):
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_size = image_size  # We'll resize input images to this size
        self.patch_size = 6  # Size of the patches to be extract from the input images
        self.num_patches = (self.image_size //self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
        
        
    def mlp(self,x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x
    
    
    
    def create_vit_classifier(self,input_shape = (128, 128, 3),num_classes = 40):
        inputs = layers.Input(shape=input_shape)
  
            # Create patches.
        patches = Patches(self.patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)
    
        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])
    
        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        self.model = keras.Model(inputs=inputs, outputs=logits)
        
        optimizer = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            )
        
        self.model.compile(
                optimizer=optimizer,
                loss="BinaryCrossentropy",
                metrics=[
                    keras.metrics.BinaryAccuracy(name="accuracy")
                ],
            )    
    
    def run_experiment(self,x_train,y_train,x_test,y_test,validation_split=0.1):
  

        checkpoint_filepath = "./model/facenet_weight.h5"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    
        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=validation_split,
            callbacks=[checkpoint_callback],
        )
    
        self.model.load_weights(checkpoint_filepath)
        _, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return history
