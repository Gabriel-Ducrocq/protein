import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA


def get_predicted_transformations(model, indexes, rotation_angles, rotation_axis, device="cpu"):
    """
    Get prediction of non amortized VAE on specific set of images given by index
    :param model: pytorch model
    :param indexes: torch.tensor (batch_size)
    :return: new struture tensor (batch_size, N_residue, 3), translations and rotation per residue
    """
    batch_size = rotation_axis.shape[0]
    proportions = torch.softmax(model.cluster_proportions, dim=1)
    log_num = -0.5 * (model.residues - model.cluster_means) ** 2 / model.cluster_std ** 2 + \
              torch.log(proportions)

    weights = torch.softmax(log_num / model.tau, dim=1)
    latent_variables = model.sample_latent(indexes)
    features = torch.cat([latent_variables, rotation_angles, rotation_axis], dim=1)
    output = model.decoder.forward(features)
    ## The translations are the first 3 scalars and quaternions the last 3
    output = torch.reshape(output, (batch_size, model.N_domains, 2 * 3))
    scalars_per_domain = output[:, :, :3]
    ones = torch.ones(size=(batch_size, model.N_domains, 1), device=device)
    quaternions_per_domain = torch.cat([ones, output[:, :, 3:]], dim=-1)
    rotations_per_residue = model.compute_rotations(quaternions_per_domain, weights)
    new_structure, translations = model.deform_structure(weights, scalars_per_domain, rotations_per_residue)
    return new_structure, translations, rotations_per_residue


def pca_latent_space(model, n_components=None):
    """

    :param model: torch model
    :return:
    """
    latent_means = model.latent_mean
    pca = PCA(n_components=n_components)
    transformed_latent_means = pca.fit_transform(latent_means.detach().numpy())
    return transformed_latent_means


def compute_mask_elements(path):
    mask = np.load(path)
    masks_n_elts = [np.sum(np.argmax(mask, axis=1) == i) for i in range(4)]
    return masks_n_elts

def mask_evolution(path):
    """
    get the evolution of the number of residues in each domain.
    :param path: string, path of foler containing all the masks
    :return: time serie of the number of residues per domain, np.array (N_iterations, N_domains)
    """
    all_files = os.listdir(path)
    mask_path = [path + "mask" + str(i) + ".npy" for i in range(400000) if "mask" + str(i) + ".npy" in all_files]
    return np.array(list(map(compute_mask_elements, mask_path)))

def plot_loss(path, title=None):
    """
    plot the loss cur
    :param path: string, path to the loss
    :return: np.array of the losses at each (batch) iteration ( NOT epoch)
    """
    loss = np.load(path)
    plt.plot(loss)
    if title:
        plt.title(title)

    plt.show()
