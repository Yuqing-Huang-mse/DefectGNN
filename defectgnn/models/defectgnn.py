import logging
import tensorflow as tf
from defectgnn.models.model_utils import swish
from defectgnn.models.initializers import GlorotOrthogonal
from defectgnn.models.layers.embedding_layer import EmbeddingLayer
from defectgnn.models.layers.gaussian_expansion import GaussianExpansion
from defectgnn.models.layers.bessel_basis_layer import BesselBasisLayer
from defectgnn.models.layers.spherical_basis_layer import SphericalBasisLayer
from defectgnn.models.layers.bond2bond_block import Bond2BondBlock
from defectgnn.models.layers.bond2atom_block import Bond2AtomBlock
from defectgnn.models.layers.atom2bond_block import Atom2BondBlock
from defectgnn.models.layers.fea_integrate_layer import FeatureIntegrationBlock
from defectgnn.models.layers.readout_layer import ReadoutLayer
from defectgnn.utils.register import registers


@registers.model.register("defectgnn")
class DefectGNN(tf.keras.Model):
    """
    DefectGNN Model
    vacancy migration GNN
    """

    def __init__(
            self,
            atom_size,
            atom_embedding_size,
            bond_embedding_size,
            hidden_size,
            num_layers,
            name='defectgnn',
            logging=None,
            cutoff=5.0,
            activation=swish,
            target_types=None,
            num_targets=1,
            kernel_initializer='zeros',
            rbf="Bessel",
            num_radial=6,
            envelope_exponent=5,
            sbf="Spherical",
            num_spherical=7,
            num_gaussian=50,
            gaussian_radial_var=0.2,
            gaussian_angular_var=0.2,
            use_extra_features=False,
            num_embedding_fc_layers=2,
            num_b2b_res_layers=2,
            num_readout_fc_layers=3,
            embedding_dropout=0,
            output_dropout=0,
            feature_add_or_concat="add",
            use_batch_norm=True,
            integrate_gate_mechanism="residual_gate",
            **kwargs
    ):
        super().__init__(name=name)
        self.logging = logging
        self.num_layers = num_layers
        self.embedding_dropout = embedding_dropout
        self.output_dropout = output_dropout

        self.target_types = target_types

        self.feature_add_or_concat = feature_add_or_concat
        if kernel_initializer == 'GlorotOrthogonal':
            kernel_initializer = GlorotOrthogonal()

        if rbf == "Bessel":
            self.rbf_layer = BesselBasisLayer(num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        elif rbf == "Gaussian":
            self.rbf_layer = GaussianExpansion(dmin=0.0, dmax=cutoff, num_gaussian=num_gaussian,
                                               var=gaussian_radial_var)
        else:
            raise ValueError("Rbf method {} isn't supported yet. We support ['Bessel', 'Gaussian'] method.".format(rbf))

        self.sbf = sbf
        if sbf == "Spherical":
            self.sbf_layer = SphericalBasisLayer(
                num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        elif sbf == "Gaussian":
            self.sbf_layer = GaussianExpansion(
                dmin=0.0, dmax=3.14, num_gaussian=num_gaussian, var=gaussian_angular_var)
        else:
            raise ValueError(
                "Sbf method {} isn't supported yet. We support ['Spherical', 'Gaussian'] method.".format(sbf))

        self.embedding_layer = EmbeddingLayer(
            atom_size, atom_embedding_size, bond_embedding_size, use_extra_features=use_extra_features,
            num_embedding_fc_layers=num_embedding_fc_layers, activation=activation,
            kernel_initializer=kernel_initializer, use_batch_norm=use_batch_norm)

        self.bond2bond_blocks = list()
        self.bond2atom_blocks = list()
        self.atom2bond_blocks = list()
        for i_layer in range(num_layers):
            self.bond2bond_blocks.append(
                Bond2BondBlock(hidden_size, num_b2b_res_layers, activation=activation,
                               kernel_initializer=kernel_initializer, use_batch_norm=use_batch_norm))
            self.bond2atom_blocks.append(
                Bond2AtomBlock(hidden_size, activation=activation, kernel_initializer=kernel_initializer,
                               use_batch_norm=use_batch_norm))

            self.atom2bond_blocks.append(
                Atom2BondBlock(hidden_size,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               use_batch_norm=use_batch_norm))

        self.feat_integrate_block = FeatureIntegrationBlock(hidden_size,
                                                           activation=activation,
                                                           use_batch_norm=True,
                                                           gate_mechanism=integrate_gate_mechanism)
        self.readout_blocks = dict()
        for target_type in self.target_types:
            self.readout_blocks[target_type] = ReadoutLayer(hidden_size,
                                                            num_readout_fc_layers,
                                                            num_targets=num_targets,
                                                            activation=activation,
                                                            kernel_initializer=kernel_initializer,
                                                            use_batch_norm=use_batch_norm)

    def call(self, inputs, training=False):
        atom_features, indices_i, indices_j = \
            inputs['atom_features_list'], inputs['id_i_list'], inputs['id_j_list']
        bond_mi_id_for_angle_mij_list, bond_ij_id_for_angle_mij_list = \
            inputs['bond_mi_id_for_angle_mij_list'], inputs['bond_ij_id_for_angle_mij_list']
        distances, angles_mij, id_swap = \
            inputs['dist_list'], inputs['angle_mij_list'], inputs['id_swap']
        vacancy_features = inputs['vacancy_features']
        for target_type in self.target_types:
            if target_type == "atom":
                reduce_to_target_indices_atom = inputs["reduce_to_target_indices_atom"]
            if target_type == "path":
                reduce_to_target_indices_path = inputs["reduce_to_target_indices_path"]

        rbf = self.rbf_layer(distances)

        if self.sbf == "Spherical":
            sbf_mij = self.sbf_layer(distances, angles_mij, bond_mi_id_for_angle_mij_list)
        elif self.sbf == "Gaussian":
            sbf_mij = self.sbf_layer(angles_mij)

        atom_embedding, bond_embedding = self.embedding_layer(atom_features, rbf, indices_i, indices_j)

        if training:
            atom_embedding = tf.nn.dropout(atom_embedding, rate=self.embedding_dropout)
            bond_embedding = tf.nn.dropout(bond_embedding, rate=self.embedding_dropout)

        atom_hidden_states = atom_embedding
        bond_hidden_states = bond_embedding

        for i_layer in range(self.num_layers):
            # Bond -> Bond
            bond_embedding = self.bond2bond_blocks[i_layer](bond_embedding,
                                                            sbf_mij,
                                                            bond_mi_id_for_angle_mij_list,
                                                            bond_ij_id_for_angle_mij_list,
                                                            id_swap)

            # Bond -> Atom
            atom_embedding = self.bond2atom_blocks[i_layer](atom_embedding, bond_embedding, indices_i, indices_j)
            atom_hidden_states = self._update_state(atom_hidden_states, atom_embedding)

            is_last_layer = ( i_layer == self.num_layers - 1 )
            needs_bond_update = not( is_last_layer and "path" not in self.target_types )

            if needs_bond_update:
                bond_embedding = self.atom2bond_blocks[i_layer](atom_embedding,
                                                                bond_embedding,
                                                                indices_i,
                                                                indices_j)
                bond_hidden_states = self._update_state(bond_hidden_states, bond_embedding)

            if training:
                atom_embedding = tf.nn.dropout(atom_embedding, rate=self.embedding_dropout)
                bond_embedding = tf.nn.dropout(bond_embedding, rate=self.output_dropout)

        outputs = dict()
        if "atom" in self.target_types:
            atom_hidden_states = self.feat_integrate_block(vacancy_features, atom_hidden_states)
            atom_outputs = self.readout_blocks["atom"](atom_hidden_states)
            outputs["atom"] = tf.gather(atom_outputs, reduce_to_target_indices_atom)
        if "path" in self.target_types:
            path_outputs = self.readout_blocks['path'](bond_hidden_states)
            outputs["path"] = tf.gather(path_outputs, reduce_to_target_indices_path)
        return outputs

    def _update_state(self, hidden_states, new_val):
        if self.feature_add_or_concat == "add":
            return hidden_states + new_val
        return tf.concat([hidden_states, new_val], axis=-1)

    def get_last_shared_layer_params(self):
        last_shared_block = self.bond2atom_blocks[-1]

        if hasattr(last_shared_block, 'dense_update'):
            return last_shared_block.dense_update.kernel

        elif hasattr(last_shared_block, 'dense'):
            return last_shared_block.dense.kernel

        elif hasattr(last_shared_block, 'W'):
            return last_shared_block.W

        else:
            trainable_vars = last_shared_block.trainable_variables
            for var in trainable_vars:
                if 'kernel' in var.name or 'weights' in var.name:
                    return var
            raise ValueError("Could not find a kernel/weight matrix in the last Bond2AtomBlock")

