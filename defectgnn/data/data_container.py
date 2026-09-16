import copy
import os
import pickle
import numpy as np
import pandas as pd
from defectgnn import registers
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

@registers.data_container.register("data_container")
class DataContainer(object):
    """
    Process original graph data
    """

    def __init__(self,
                 graph_data_list,
                 targets_data_list,
                 target_normalizer=None,
                 target_types=None,
                 target_col="targets",
                 normalize_labels=False,
                 raw_graph_data_list=None,
                 raw_structure_list=None,
                 atom_feature_scheme=None,
                 ):
        self.graph_data_list = np.array(graph_data_list)
        self.raw_graph_data_list = raw_graph_data_list
        self.raw_structure_list = raw_structure_list
        self.target_types = target_types
        self.atom_feature_scheme = atom_feature_scheme

        # define the atom feature scheme
        atom_features = graph_data_list[0]["atom_features_list"]
        self.atom_feature_len = atom_features.shape[-1]

        # if isinstance(atom_features[0], (np.int32, np.int64, int)):
        #     self.atom_feature_scheme = "specie_onehot"
        # elif isinstance(atom_features[0], (list, np.ndarray, pd.Series, tuple)):
        #     self.atom_feature_scheme = "external"
        #     self.atom_feature_len = atom_features.shape[-1]
        # else:
        #     raise ValueError("Cannot determine atom_feature_scheme", atom_features[0])

        # targets
        self.target_data_list = np.array(targets_data_list)
        self.target_col = target_col

        # normalization
        # before normalization, must do data cleaning
        if normalize_labels:
            if target_normalizer is None:
                self.target_normalizer = dict()
                target_values = defaultdict(list)

                for target_dict in self.target_data_list:
                    targets = target_dict[self.target_col]
                    for target_type in self.target_types:
                        target_values[target_type].extend(targets[target_type])

                for target_type in self.target_types:
                    scaler = StandardScaler()
                    data = np.array(target_values[target_type]).reshape(-1,1)
                    scaler.fit(data)
                    self.target_normalizer[target_type] = scaler
            else:
                self.target_normalizer = target_normalizer
        else:
            self.target_normalizer = None

    @classmethod
    def from_files(
            cls,
            graph_data_file_list,
            targets_data_file,
            target_normalizer=None,
            target_col="targets",
            target_types=None,
            normalize_labels=True,
            atom_feature_scheme=None,
    ):
        # graph data
        graph_data_list = list()
        for file_idx, graph_data_file in enumerate(graph_data_file_list):
            try:
                with open(graph_data_file[0],'rb') as f:
                    graph_data = pickle.load(f)
                    graph_data_list.append(graph_data)
            except Exception:
                raise ValueError("Please provide paths to the pickled graph_data, "
                                 "and make sure the files can be pickle.load")

        # target data
        try:
            with open(targets_data_file, "rb") as f:
                target_data_list = pickle.load(f)
        except Exception:
            raise ValueError("Please provide paths to the pickled target_data_list, "
                             "and make sure the files can be pickle.load")

        # create vacancy graph data from original complete graph data
        raw_structure_list = []
        vacancy_graph_data_list = []
        vacancy_target_data_list = []

        VACANCY_ID = 'vacancy_id'
        TARGETS = 'targets'
        BOND_ID = 'bond_id'
        NEIGHBOR_ID = 'neighbor_id'

        for structure_idx, (graph_data_dict, target_data_dict) in enumerate(zip(graph_data_list, target_data_list)):
            id_i_list = graph_data_dict['id_i_list']
            id_j_list = graph_data_dict['id_j_list']
            atom_features_list = graph_data_dict['atom_features_list']

            reduce_to_target_indices = target_data_dict['reduce_to_target_indices']
            targets = target_data_dict[target_col]

            targets_df = {}
            for target_type in target_types:
                if target_type == "atom":
                    df = pd.DataFrame({
                        VACANCY_ID: reduce_to_target_indices[target_type],
                        TARGETS: targets[target_type]
                    })
                    df_sorted = df.sort_values(by=[VACANCY_ID]).reset_index(drop=True)
                elif target_type == "path":
                    df = pd.DataFrame({
                        BOND_ID: reduce_to_target_indices[target_type],
                        TARGETS: targets[target_type],
                        VACANCY_ID: id_i_list[reduce_to_target_indices[target_type]],
                        NEIGHBOR_ID: id_j_list[reduce_to_target_indices[target_type]]
                    })
                    df_sorted = df.sort_values(by=[VACANCY_ID, NEIGHBOR_ID]).reset_index(drop=True)
                else:
                    raise KeyError("Target type only can choose from 'atom' and 'path'")
                targets_df[target_type] = df_sorted


            ref_df_sorted = targets_df[target_types[0]]


            unique_vacancy_ids = ref_df_sorted[VACANCY_ID].unique()

            for i in unique_vacancy_ids:
                should_skip = False
                empty_target_types = []

                for target_type in target_types:
                    df_sorted = targets_df[target_type]
                    df_i = df_sorted[df_sorted[VACANCY_ID] == i]

                    if df_i.empty:
                        should_skip = True
                        empty_target_types.append(target_type)

                        print(
                            f"Warning: structure {structure_idx}, vacancy_id {i} "
                            f"has no {target_type} target data")


                if should_skip:

                    continue

                raw_structure_list.append(structure_idx)



                new_atom_features_list = atom_features_list.copy()

                vacancy_feature = np.zeros_like(new_atom_features_list[i])
                vacancy_feature[-1] = 1
                # new_atom_features_list[i] = np.zeros_like(atom_features_list[i])
                new_atom_features_list[i] = vacancy_feature


                new_graph_data_dict = graph_data_dict.copy()
                new_graph_data_dict['atom_features_list'] = new_atom_features_list


                if not new_graph_data_dict.get('vacancy_features'):

                    original_atom_feature = atom_features_list[i]
                    vacancy_features = [np.zeros_like(feat) for feat in atom_features_list]
                    vacancy_features[i] = original_atom_feature
                    new_graph_data_dict['vacancy_features'] = vacancy_features
                else:

                    new_graph_data_dict['vacancy_features'] = new_graph_data_dict['vacancy_features'].get(i)


                indices_dict = {}
                targets_dict = {}

                for target_type in target_types:
                    df_sorted = targets_df[target_type]
                    df_i = df_sorted[df_sorted[VACANCY_ID] == i].reset_index(drop=True)

                    if target_type == "atom":
                        indices_dict[target_type] = df_i[VACANCY_ID].to_numpy(dtype=np.int32)
                        targets_dict[target_type] = df_i[TARGETS].to_numpy(dtype=np.float32)
                    elif target_type == "path":
                        indices_dict[target_type] = df_i[BOND_ID].to_numpy(dtype=np.int32)
                        targets_dict[target_type] = df_i[TARGETS].to_numpy(dtype=np.float32)

                new_target_data_dict = {
                    "reduce_to_target_indices": indices_dict,
                    "targets": targets_dict
                }

                vacancy_graph_data_list.append(new_graph_data_dict)
                vacancy_target_data_list.append(new_target_data_dict)

        return cls(graph_data_list=vacancy_graph_data_list,
                   targets_data_list=vacancy_target_data_list,
                   target_normalizer=target_normalizer,
                   target_types=target_types,
                   target_col=target_col,
                   normalize_labels=normalize_labels,
                   raw_graph_data_list=graph_data_list,
                   raw_structure_list=raw_structure_list,
                   atom_feature_scheme=atom_feature_scheme,)


    @classmethod
    def from_structures(
            cls,
            structure_list=None,
            vacancy_feature_list=None,
            targets_dict=None,
            source_ids=None,
            target_types=None,
            neighbor_scheme='pmg_cutoff',
            neighbor_cutoff=4.5,
            external_neighbors_list=None,
            bond_distances_list=None,
            atom_feature_scheme="specie_onehot",
            external_atom_features_list=None,
            specie_to_features=None,
            output_graph_data=True,
            output_targets=True,
            save_graph_data_batch_size=1,
            output_path=None
    ):

        def create_onehot_vector(indice, num_classes):
            return np.eye(num_classes)[indice].astype(np.int32)

        batch_graph_data_list = list()
        total_graph_data_list = list()
        total_target_data_list = list()
        batch_id = 0

        # enumerate the structure list
        for structure_id, structure in enumerate(structure_list):

            if os.path.isfile(os.path.join(output_path, f"graph_data_{structure_id}.pkl")):
                continue

            print("Generating graph data for structure {}".format(structure_id + 1))
            atom_features_list = list()
            dist_list = list()
            map_ij_to_bond_id = dict()

            id_i_list = list()
            id_j_list = list()

            angle_mij_list = list()
            bond_mi_id_for_angle_mij_list = list()
            bond_ij_id_for_angle_mij_list = list()

            bond_id = 0
            nn_infos = list()
            if neighbor_scheme == "external":
                for i in range(len(structure)):
                    i_coords = structure[i].coords

                    js = external_neighbors_list[structure_id][i]

                    js_to_i_frac = structure.frac_coords[js] - structure[i].frac_coords
                    js_cross_pbc = np.abs(np.round(js_to_i_frac)).sum(axis=1) > 0

                    js_to_i_frac_image = js_to_i_frac - np.round(js_to_i_frac)
                    js_frac_coords = structure[i].frac_coords + js_to_i_frac_image

                    js_coords = np.dot(js_frac_coords, structure.lattice.matrix)

                    if bond_distances_list is None:
                        dists = np.linalg.norm(js_coords - i_coords, axis=1)
                    else:
                        dists = bond_distances_list[structure_id][i]

                    nn_infos.append({"nn_ids": js,
                                     "nn_coords_image": js_coords,
                                     "nn_frac_coords_image": js_frac_coords,
                                     "nn_dists": dists,
                                     "nn_cross_pbc": js_cross_pbc,
                                     "n_neighbors": len(js)})

            elif neighbor_scheme == "pmg_cutoff":
                for i in range(len(structure)):
                    nns = structure.get_neighbors(structure[i], r=neighbor_cutoff)
                    nn_infos.append({"nn_ids": [nn[2] for nn in nns],
                                     "nn_coords_image": [nn[0].coords for nn in nns],
                                     "nn_sites_image": [nn[0] for nn in nns],
                                     "nn_dists": [nn[1] for nn in nns],
                                     "nn_cross_pbc": [True if np.abs(nn[3]).sum() != 0 else False for nn in nns],
                                     "n_neighbors": len(nns)})

            else:
                raise ValueError("No support for neighbor_scheme = {}".format(neighbor_scheme))

            # enumerate atoms in structure
            for i in range(len(structure)):
                # i's coords and features
                i_coords = structure[i].coords

                if atom_feature_scheme == "external":
                    i_feature = external_atom_features_list[structure_id][i]
                elif atom_feature_scheme == "specie_onehot":
                    i_feature = specie_to_features[structure.sites[i].specie.symbol]

                    i_feature = create_onehot_vector(i_feature,len(specie_to_features) + 1)
                else:
                    raise ValueError("No support for atom_feature_scheme = {}".format(atom_feature_scheme))

                atom_features_list.append(i_feature)

                # iterate neighbors of i, namely j
                nn_info = nn_infos[i]
                for idx in range(nn_info["n_neighbors"]):
                    j = nn_info["nn_ids"][idx]
                    j_coords = nn_info["nn_coords_image"][idx]
                    dist = nn_info["nn_dists"][idx]

                    map_ij_to_bond_id["{}_{}".format(i, j)] = bond_id

                    id_i_list.append(i)
                    id_j_list.append(j)

                    dist_list.append(dist)

                    # m: neighbors of i
                    ms = nn_infos[i]["nn_ids"]
                    m_coords = nn_infos[i]["nn_coords_image"]
                    for m, m_coords in zip(ms, m_coords):
                        if (m != j):
                            # Angle: im vs ij
                            vector1 = j_coords - i_coords
                            vector2 = m_coords - i_coords
                            angle_mij = DataContainer._calculate_neighbor_angles(vector1, vector2)
                            angle_mij_list.append(angle_mij)

                            bond_mi_id_for_angle_mij_list.append("{}_{}".format(m, i))
                            bond_ij_id_for_angle_mij_list.append(bond_id)

                    bond_id += 1

            bond_mi_id_for_angle_mij_list = list(map(lambda x: map_ij_to_bond_id[x], bond_mi_id_for_angle_mij_list))

            # initialize id_swap as length of dist_list
            id_swap = [None] * len(dist_list)

            # create a dict to store the index of (ij)
            index_dict = {(i, j): idx for idx, (i, j) in enumerate(zip(id_i_list, id_j_list))}

            # find the swapped index for (j, i), here we require both i-j and j-i are included.
            for index, (i, j) in enumerate(zip(id_i_list, id_j_list)):
                id_swap[index] = index_dict.get((j, i))

            # deal with targets
            targets = dict()
            reduce_to_target_indices = dict()

            for target_type, targets_list in targets_dict.items():
                is_broken = False
                try:
                    target = targets_list[structure_id]
                except IndexError:
                    is_broken = True
                    break

                if target_type == "atom":
                    if isinstance(target, dict):
                        reduce_to_target_indices[target_type] = list(target.keys())
                        targets[target_type] = list(target.values())
                    elif isinstance(target, (list, np.ndarray, tuple, pd.Series)):
                        reduce_to_target_indices = np.arange(len(structure))
                    else:
                        raise ValueError("Targets_list of atom-level dataset: "
                                         "Only supports list of lists or list of dicts")

                elif target_type == "path":
                    if isinstance(target, dict):
                        bond_indices = ["{}_{}".format(x, y) for x, y in target.keys()]
                        reduce_to_target_indices[target_type] = list(map(lambda x: map_ij_to_bond_id[x], bond_indices))
                        targets[target_type] = list(target.values())
                    else:
                        raise ValueError("Targets_list of path-level dataset: "
                                         "Only supports list of dicts, with the key as tuple of bond indices")

            if not is_broken:
                # vacancy_feature
                if vacancy_feature_list:
                    vacancy_features_dict = vacancy_feature_list[structure_id]
                else:
                    vacancy_features_dict = None

                graph_data_dict = {
                    "atom_features_list": np.array(atom_features_list).astype(
                        np.float32 if atom_feature_scheme == "external" else np.int32),
                    "dist_list": np.array(dist_list).astype(np.float32),
                    "id_i_list": np.array(id_i_list).astype(np.int32),
                    "id_j_list": np.array(id_j_list).astype(np.int32),
                    "angle_mij_list": np.array(angle_mij_list).astype(np.float32),
                    "bond_mi_id_for_angle_mij_list": np.array(bond_mi_id_for_angle_mij_list).astype(np.int32),
                    "bond_ij_id_for_angle_mij_list": np.array(bond_ij_id_for_angle_mij_list).astype(np.int32),
                    "id_swap": np.array(id_swap).astype(np.int32),
                    "n_structures": 1,
                    "vacancy_features":vacancy_features_dict,
                }

                targets_data_dict = {
                    "reduce_to_target_indices": {k: np.array(v).astype(np.int32) for k, v in
                                                 reduce_to_target_indices.items()},
                    "targets": {k: np.array(v).astype(np.float32 ) for k, v in targets.items()}}


                total_graph_data_list.append(graph_data_dict)
                total_target_data_list.append(targets_data_dict)

                if output_graph_data and save_graph_data_batch_size is not None:
                    os.makedirs(output_path,exist_ok=True)

                    # save the pkl file of each graph sample to file. No need to recalculate anymore.
                    if save_graph_data_batch_size == 1:
                        graph_data_id = source_ids[structure_id] if source_ids is not None else structure_id
                        with open(os.path.join(output_path, "graph_data_{}.pkl".format(graph_data_id)), "wb") as f:
                            pickle.dump(graph_data_dict, f, protocol=4)

                    else:
                        batch_graph_data_list.append(graph_data_dict)
                        # Default: drop last
                        if (structure_id + 1) % save_graph_data_batch_size == 0:
                            with open(os.path.join(output_path, "graph_data_batchsize{}_{}.pkl".format(
                                    save_graph_data_batch_size, batch_id)), "wb") as f:
                                pickle.dump(batch_graph_data_list, f, protocol=4)
                            batch_id += 1
                            batch_graph_data_list = list()

        if output_graph_data and save_graph_data_batch_size == len(structure_list):
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "graph_data.pkl"), "wb") as f:
                pickle.dump(total_graph_data_list, f, protocol=4)

        if output_targets:
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "target_data.pkl"), "wb") as f:
                pickle.dump(total_target_data_list, f, protocol=4)

        return cls(graph_data_list=total_graph_data_list,
                   target_types=target_types,
                   targets_data_list=total_target_data_list,
                   atom_feature_scheme=atom_feature_scheme,
                  )

    @staticmethod
    def collate_pool(graph_data_slice, target_data_slice,
                     target_types=None,
                     target_normalizer=None,
                     target_col="targets"):

        n_structures = len(graph_data_slice)
        atom_features_list = list()
        dist_list = list()
        angle_mij_list = list()
        id_i_list = list()
        id_j_list = list()
        bond_mi_id_for_angle_mij_list = list()
        bond_ij_id_for_angle_mij_list = list()
        id_swap_list = list()
        vacancy_features_list = list()

        reduce_to_target_indices = defaultdict(list)
        targets = defaultdict(list)

        atom_num = 0
        bond_num = 0

        for idx, (graph_data_dict, target_data_dict) in enumerate(zip(graph_data_slice, target_data_slice)):
            atom_features_list.append(graph_data_dict["atom_features_list"])
            dist_list.append(graph_data_dict["dist_list"])
            angle_mij_list.append(graph_data_dict["angle_mij_list"])
            id_i_list.append(graph_data_dict["id_i_list"] + atom_num)
            id_j_list.append(graph_data_dict["id_j_list"] + atom_num)
            bond_mi_id_for_angle_mij_list.append(graph_data_dict["bond_mi_id_for_angle_mij_list"] + bond_num)
            bond_ij_id_for_angle_mij_list.append(graph_data_dict["bond_ij_id_for_angle_mij_list"] + bond_num)
            id_swap_list.append(graph_data_dict["id_swap"] + bond_num)
            vacancy_features_list.append(graph_data_dict['vacancy_features']) 

            unprocessed_reduce_to_target_indices = target_data_dict["reduce_to_target_indices"]
            for target_type in target_types:

                if target_type == "atom":
                    reduce_to_target_indices[target_type].append(
                        unprocessed_reduce_to_target_indices[target_type] + atom_num)
                elif target_type == "path":
                    reduce_to_target_indices[target_type].append(
                        unprocessed_reduce_to_target_indices[target_type] + bond_num)
                targets[target_type].append(target_data_dict[target_col][target_type])

            atom_num += len(graph_data_dict["atom_features_list"])
            bond_num += len(graph_data_dict["dist_list"])

        # return a dict，containing input_5000 and target
        input_target_data_dict_for_slice = {
            "atom_features_list": np.concatenate(atom_features_list),
            "dist_list": np.concatenate(dist_list),
            "angle_mij_list": np.concatenate(angle_mij_list),
            "id_i_list": np.concatenate(id_i_list),
            "id_j_list": np.concatenate(id_j_list),
            "bond_mi_id_for_angle_mij_list": np.concatenate(bond_mi_id_for_angle_mij_list),
            "bond_ij_id_for_angle_mij_list": np.concatenate(bond_ij_id_for_angle_mij_list),
            "id_swap": np.concatenate(id_swap_list),
            "n_structures": n_structures,
            "reduce_to_target_indices": {k: np.concatenate(v) for k, v in reduce_to_target_indices.items()},
            "vacancy_features": np.concatenate(vacancy_features_list,axis=0),
        }

        targets_data_dict = {}
        for target_type, target_data_list in targets.items():
            if target_type != "structure":
                target_data_list = np.concatenate(target_data_list)
            else:
                target_data_list = np.array(target_data_list)
            targets_data_dict[target_type] = target_data_list

        # transform targets
        for target_type in target_types:
            if len(targets_data_dict[target_type].shape) == 1:
                targets_data_dict[target_type] = targets_data_dict[target_type][:, np.newaxis]
            if target_normalizer:
                targets_data_dict[target_type] = target_normalizer[target_type].transform(targets_data_dict[target_type])

        input_target_data_dict_for_slice["targets"] = targets_data_dict

        return input_target_data_dict_for_slice
    
    @staticmethod
    def _calculate_neighbor_angles(vector1, vector2):
        x = np.sum(vector1 * vector2, axis=-1)
        y = np.cross(vector1, vector2)
        y = np.linalg.norm(y, axis=-1)
        angle = np.arctan2(y, x)
        return angle

    def __len__(self):
        if self.raw_graph_data_list is not None:
            return len(self.raw_graph_data_list)
        return len(self.graph_data_list)

    @staticmethod
    def int_keys():
        return ['id_i_list', 'id_j_list',
                'bond_mi_id_for_angle_mij_list', 'bond_ij_id_for_angle_mij_list',
                 "id_swap"]

    @staticmethod
    def float_keys():
        return ['dist_list', 'angle_mij_list']

    @staticmethod
    def int_number_keys():
        return ["n_structures"]

    def __getitem__(self, idx):
        return self.collate_pool(
            graph_data_slice=self.graph_data_list[idx],
            target_data_slice=self.target_data_list[idx],
            target_types=self.target_types,
            target_normalizer=self.target_normalizer,
            target_col=self.target_col
        )



