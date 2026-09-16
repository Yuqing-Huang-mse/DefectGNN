import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from defectgnn.models.layers.residual_layer import ResidualLayer


class Bond2BondBlock(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_b2b_res_layers,
        name='bond2bond_block',
        activation=None,
        kernel_initializer="glorot_uniform",
        use_batch_norm=True,
        enable_id_swap=True
    ):
        super().__init__(name=name)
        self.enable_id_swap = enable_id_swap

        if use_batch_norm:
            self.bond_kj_fc_layers = Sequential([
                layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
                layers.BatchNormalization(),
                layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
                layers.BatchNormalization()])

            self.bond_preprocess_fc_layer_ij = Sequential([
                layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
                layers.BatchNormalization()])

            self.bond_preprocess_fc_layer_ji = Sequential([
                layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
                layers.BatchNormalization()])
        else:
            self.bond_kj_fc_layers = Sequential([
                layers.Dense(hidden_size + 1, activation=activation, use_bias=False,
                             kernel_initializer=kernel_initializer),
                layers.Dense(hidden_size + 1, activation=activation, use_bias=False,
                             kernel_initializer=kernel_initializer)
            ])
            self.bond_preprocess_fc_layer_ij = layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer)
            self.bond_preprocess_fc_layer_ji = layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer)

        self.angle_ijk_attention_layers = Sequential([
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer)])

        self.residual_layers = list()
        for i in range(num_b2b_res_layers):
            self.residual_layers.append(ResidualLayer(
                hidden_size, activation=activation, use_bias=True, use_batch_norm=use_batch_norm,
                kernel_initializer=kernel_initializer, name="b2b_residual_layer_{}".format(i)))
        self.residual_layers = Sequential(self.residual_layers)

    def call(self, bond_embedding, sbf_mij, bond_mi_id_for_angle_mij_list, bond_ij_id_for_angle_mij_list, id_swap):

        bond_embedding_mij_kj = tf.gather(bond_embedding, bond_mi_id_for_angle_mij_list)
        bond_embedding_mij_ij = tf.gather(bond_embedding, bond_ij_id_for_angle_mij_list)
        bond_mij_updated = self.bond_kj_fc_layers(tf.concat([bond_embedding_mij_kj, bond_embedding_mij_ij], axis=-1))
        angle_mij_attentions = self.angle_ijk_attention_layers(sbf_mij)

        num_bonds = tf.shape(bond_embedding)[0]
        bond_embedding_b2b = tf.math.unsorted_segment_sum(angle_mij_attentions * bond_mij_updated[:, :1] * bond_mij_updated[:, 1:], bond_ij_id_for_angle_mij_list, num_bonds)

        bond_update = self.bond_preprocess_fc_layer_ij(bond_embedding_b2b)
        if self.enable_id_swap:
            bond_update += tf.gather(
                self.bond_preprocess_fc_layer_ji(bond_embedding_b2b), id_swap)
        bond_embedding += bond_update
        bond_embedding = self.residual_layers(bond_embedding)
        return bond_embedding

