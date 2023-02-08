import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.vectors import Vector
from scipy.spatial.transform import Rotation


def get_positions(residue, name):
    x = residue["CA"].get_coord()
    y = residue["N"].get_coord()
    if name == "GLY":
        z = residue["C"].get_coord()
        return x.copy(),y.copy(),z.copy()

    z = residue["C"].get_coord()
    return x.copy(),y.copy(),z.copy()

def norm(u):
    """
    Computes the euclidean norm of a vector
    :param u: vector
    :return: euclidean norm of u
    """
    return np.sqrt(np.sum(u**2))

def gram_schmidt(u1, u2):
    """
    Orthonormalize a set of two vectors.
    :param u1: first non zero vector, unnormalized
    :param u2: second non zero vector, unormalized
    :return: orthonormal basis
    """
    e1 = u1/norm(u1)
    e2 = u2 - np.dot(u2, e1)*e1
    e2 /= norm(e2)
    return e1, e2

def get_orthonormal_basis(u1, u2):
    """
    Computes the local orthonormal frame basis based on the Nitrogen, C alpha and C beta atoms.
    :param u1: first non zero vector, unnormalized (here the bond vector between Nitrogen and C alpha carbon)
    :param u2: second non zero vector, unormalized  (here the bond vector between C beta and C alpha carbon)
    :return: An array of three orthonormal vectors of size 3, in colums
    """

    e1, e2 = gram_schmidt(u1, u2)
    e3 = np.cross(e1, e2)
    return np.array([e1, e2, e3]).T


def rotate_pdb_structure_axis_angle(pdb_structure, axis, angle):
    """

    :param axis: numpy.array (1, 3) of axis against which we make the rotation
    :param angle: float, angle of the rotation in radians
    """
    assert np.sum(axis) == 1
    rotvec = angle*axis
    rotation = Rotation.from_rotvec(rotvec)
    rotation_matrix = rotation.as_matrix()
    all_coords = np.concatenate([atom.coord[:, None] for atom in pdb_structure.get_atoms()], axis=1)
    rotated_coordinates = np.matmul(rotation_matrix[0], all_coords)
    for index, atom in enumerate(pdb_structure.get_atoms()):
        atom.set_coord(rotated_coordinates[:, index])


def rotate_pdb_structure_matrix(pdb_structure, rotation_matrix):
    """

    :param axis: numpy.array (1, 3) of axis against which we make the rotation
    :param angle: float, angle of the rotation in radians
    """
    all_coords = np.concatenate([atom.coord[:, None] for atom in pdb_structure.get_atoms()], axis=1)
    rotated_coordinates = np.matmul(rotation_matrix[0], all_coords)
    for index, atom in enumerate(pdb_structure.get_atoms()):
        atom.set_coord(rotated_coordinates[:, index])

def translate_pdb_structure(pdb_structure, translation):
    """

    :param pdb_structure: pdb structrue in Biopython
    :param translation: np.array (1, 3)
    :return:
    """
    for index, atom in enumerate(pdb_structure.get_atoms()):
        atom.set_coord(atom.coord + translation)

def rotate_domain_pdb_structure_axis_angle(pdb_structure, start_residue, end_residue, axis, angle, local_frame):
    """
    rotates a domain of the protein only expressed in a specific frame of reference
    :param pdb_structure: pdb structure in biopython
    :param start_residue: integer, the (included) starting residue number of the domain
    :param end_residue: integer, the (excluded) ending residue number of the domain
    :param axis: numpy.array (1, 3) of axis against which we make the rotation
    :param angle: angle of the rotation in radians
    :param local_frame: np.array(3, 3) of basis vector of norm 1 in columns
    """
    assert np.sum(axis) == 1.0
    assert (np.sum(np.round(np.sum(local_frame**2, axis=0), 3) == 1.0)).all() == True
    rotvec = angle*axis
    rotation = Rotation.from_rotvec(rotvec)
    rotation_matrix = rotation.as_matrix()
    rotated_frame = np.matmul(rotation_matrix, local_frame)
    for idx, res in enumerate(pdb_structure.get_residues()):
        if idx >= start_residue and idx < end_residue:
            for atom in res.get_atoms():
                local_coordinates = np.matmul(atom.coord, local_frame)
                rotated_coord = np.matmul(local_coordinates, rotated_frame.T)
                atom.set_coord(rotated_coord)


def rotate_domain_pdb_structure_matrix(pdb_structure, start_residue, end_residue, rotation_matrix, local_frame):
    """
    rotates a domain of the protein only expressed in a specific frame of reference
    :param pdb_structure: pdb structure in biopython
    :param start_residue: integer, the (included) starting residue number of the domain
    :param end_residue: integer, the (excluded) ending residue number of the domain
    :param rotation_matrix: np.array(3,3) rotation matrix.
    :param local_frame: np.array(3, 3) of basis vector of norm 1 in columns
    """
    rotated_frame = np.matmul(rotation_matrix, local_frame)
    for idx, res in enumerate(pdb_structure.get_residues()):
        if idx >= start_residue and idx < end_residue:
            for atom in res.get_atoms():
                local_coordinates = np.matmul(atom.coord, local_frame)
                rotated_coord = np.matmul(local_coordinates, rotated_frame.T)
                atom.set_coord(rotated_coord)


def rotate_residues(pdb_structure, rotation_matrix, local_frame):
    """
    Rotate the residues with corresponding rotation matrices.
    :param pdb_structure: pdb structure in biopython
    :param rotation_matrix: np array (N_residue, 3,3) of rotation matrices per residue
    :param local_frame: np.array(3, 3) of basis vector of norm 1 in columns
    """
    for idx, res in enumerate(pdb_structure.get_residues()):
        rotated_frame = np.matmul(rotation_matrix[idx], local_frame)
        for atom in res.get_atoms():
            local_coordinates = np.matmul(atom.coord, local_frame)
            rotated_coord = np.matmul(local_coordinates, rotated_frame.T)
            atom.set_coord(rotated_coord)

def translate_residues(pdb_structure, translations):
    """
    Translate the residues with the corresponding translation vector
    :param pdb_structure: pdb structure in biopython
    :param translations: np array (N_residue,3) of rotation matrices per residue
    """
    for idx, res in enumerate(pdb_structure.get_residues()):
        for atom in res.get_atoms():
            atom.set_coord(atom.coord + translations[idx])



def translate_domain_pdb_structure(pdb_structure, start_residue, end_residue, translation):
    """

    :param pdb_structure: pdb structure in biopython
    :param start_residue: integer, the (included) starting residue number of the domain
    :param end_residue: integer, the (excluded) ending residue number of the domain
    :param translation: numpy.array (1, 3) translation vector
    """
    for idx, res in enumerate(pdb_structure.get_residues()):
        if idx >= start_residue and idx < end_residue:
            for atom in res.get_atoms():
                atom.set_coord(atom.coord + translation)


def compute_rmsd_pdb(path1, path2):
    """
    Compute the rmsd between two structures, given by the .pdb files at path1 and path2
    The two structures must represent the SAME protein with different possible conformations,
     and they are NOT superimposed first !!!
    :param path1: str, path to the first .pdb file
    :param path2: str, path to the second .pdb file
    :return: float, the rmsd between the two structures
    """
    list_distances = []
    parser = PDBParser()
    pdb_structure1 = parser.get_structure("1", path1)
    pdb_structure2 = parser.get_structure("2", path2)
    for res1, res2 in zip(pdb_structure1.get_residues(),pdb_structure2.get_residues()):
        for atom1, atom2 in zip(res1.get_atoms(), res2.get_atoms()):
            list_distances.append(np.sum((atom1.coord - atom2.coord)**2))

    return np.mean(list_distances)




def project_image(pdb_structure):
    """
    Project the structure on the x-y plane
    :param pdb_structure: pdb_structure in biopython
    """
    pass


"""
parser = PDBParser()
pdb_structure = parser.get_structure("1", "../ranked_0.pdb")
first_res = list(pdb_structure.get_residues())[0]
name_first_res = first_res.get_resname()
x, y, z = get_positions(first_res, name_first_res)
local_frame_in_col = get_orthonormal_basis(y-x, z-x)

#rotate_domain_pdb_structure(pdb_structure, 0, 300, np.array([0, 0, 1]), -np.pi/4, local_frame_in_col)
translate_domain_pdb_structure(pdb_structure, 0, 300, np.array([-8, -8, 0]))
rotate_domain_pdb_structure(pdb_structure, 1353, 1511, np.array([0, 1, 0]), -np.pi/2, local_frame_in_col)

#translate_domain_pdb_structure(pdb_structure, 1000, 1511, np.array([5, 5, 0]))
io = PDBIO()
io.set_structure(pdb_structure)
io.save("../out2.pdb")

(local_frame_in_col)
"""




