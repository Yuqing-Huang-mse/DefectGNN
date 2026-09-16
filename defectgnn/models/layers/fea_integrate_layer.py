"""
Feature integration for vacancy-site information.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class FeatureIntegrationBlock(layers.Layer):
    def __init__(
            self,
            hidden_size,
            name='feature_integration',
            activation='relu',
            kernel_initializer='glorot_uniform',
            use_batch_norm=True,
            gate_mechanism='residual_gate',
            physics_constraints=None):

        super().__init__(name=name)
        self.gate_mechanism = gate_mechanism
        self.physics_constraints = physics_constraints

        # feature_projection layer
        self.feature_projection = Sequential([
            layers.Dense(hidden_size,
                         activation=activation,
                         kernel_initializer=kernel_initializer)
        ])

        if use_batch_norm:
            self.feature_projection.add(layers.BatchNormalization())


        if gate_mechanism == 'residual_gate':
            self.gate_layer = layers.Dense(
                hidden_size,
                activation='sigmoid',
                kernel_initializer=kernel_initializer,
            )

        elif gate_mechanism == 'attention_gate':
            pass

        elif gate_mechanism == 'adaptive_gate':
            pass

        elif gate_mechanism == 'physics_gate' and physics_constraints:
            pass

    def call(self, vacancy_features, hidden_states):


        projected_features = self.feature_projection(vacancy_features)


        if self.gate_mechanism == None:
            return hidden_states + projected_features

        elif self.gate_mechanism == 'residual_gate':

            gate_values = self.gate_layer(tf.concat([hidden_states, projected_features], axis=-1))
            return gate_values * hidden_states + (1 - gate_values) * projected_features

        elif self.gate_mechanism == 'attention_gate':
            pass

        elif self.gate_mechanism == "adaptive_gate":
            pass

        elif self.gate_mechanism == 'feature_cross':
            pass

        elif self.gate_mechanism == 'physics_gate' and self.physics_constraints:
            pass

        else:
            return hidden_states + projected_features

