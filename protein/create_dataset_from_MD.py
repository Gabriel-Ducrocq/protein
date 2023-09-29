import os
import numpy as np
from copy import deepcopy
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.vectors import Vector
from scipy.spatial.transform import Rotation
from protein.main import rotate_pdb_structure_axis_angle


# This module creates


def generate_poses(n_poses=10):
    """

    :param n_poses: integer, number of poses
    :return: np.array(n_poses), np.array(n_poses, 3), arrays of rotation angle and rotation axis.
    """
    axis_rotation = np.random.normal(size=(n_poses, 3))
    axis_rotation /= np.sqrt(np.sum(axis_rotation ** 2, axis=-1))[:, None]
    angle_rotation = np.random.uniform(low=0, high=np.pi, size=n_poses)
    return angle_rotation, axis_rotation


def generate_different_poses_one_image(structure, n_poses=10):
    """
    This function takes a protein structure as a BioPDB structure and generates different poses of it.
    :param structure: BioPDB structure of the protein
    :param n_poses: integer, number of different pose to generate for this structure
    :return: BioPDB structure, rotated structure
    """
    angle_rotation, axis_rotation = generate_poses(n_poses)
    list_rotated_structures = []
    for i in range(n_poses):
        copy_structure = deepcopy(structure)
        rotate_pdb_structure_axis_angle(copy_structure, axis_rotation[i][None, :], angle_rotation[i])
        list_rotated_structures.append(copy_structure)

    return list_rotated_structures, angle_rotation, axis_rotation


def generate_different_poses(list_structures_path, list_n_poses, results_path):
    """

    :param list_structures_path: list of strings, path to the PDB structures
    :param list_n_poses: list of integer, number of poses to generate for each structure
    :return: None
    """
    parser = PDBParser()
    for n_poses, structure_path, results_path in zip(list_n_poses, list_structures_path, results_path):
        pdb_structure = parser.get_structure("1", structure_path)
        list_rotated_structure, angle_rotation, axis_rotation = generate_different_poses_one_image(pdb_structure,
                                                                                                   n_poses=n_poses)

        structure_name = structure_path.split("/")[-1].split(".")[0]
        os.mkdir(results_path)
        os.mkdir(results_path + f"{structure_name}/")
        np.save(results_path + f"{structure_name}/" + "angle_rotation.npy", angle_rotation)
        np.save(results_path + f"{structure_name}/" + "axis_rotation.npy", axis_rotation)
        for i in range(n_poses):
            struct = list_rotated_structure[i]
            io = PDBIO()
            io.set_structure(struct)
            io.save(results_path + f"{structure_name}/{structure_name}{i}.pdb")

