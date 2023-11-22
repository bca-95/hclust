#!/usr/bin/env python3.7
# edit
import numpy as np
import pandas as pd
import random

import MDAnalysis as mda

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt

"""
This module aims to preprocess the data, as data subsampling, setting of the features, 
dimensionality reduction, and the plotting of the transformed data.
"""


def subsample(u, percentage_subsample=False) :
    """Subsampling data
    INPUTS : 
    ------
    u : mda Universe
    percentage_subsample (False, default) : percentage of data to keep
        If False, no subsampling is applied on the data
        If a float is giving (between 1 and 100), it corresponds to the percentage of data to be keepin
    """
    len_data = u.trajectory.n_frames

    if isinstance(percentage_subsample, float) : 
        if not 1 <= percentage_subsample <= 100:
            raise ValueError("percentage should be between 1 and 100")
        size = int((len_data * percentage_subsample)/100)
        index_sel = np.random.randint(0, len_data, size)

    if percentage_subsample == False :
        index_sel = np.arange(len_data)
    print("The data has {} points".format(len(index_sel)))
    return index_sel


def features_selection(pdb, traj, features, return_features="protein", percentage_subsample=False) :
    """ ---  This function sets the features used for the clusterization  ---
    The package mdAnalysis is performed in order to select atoms of interest, used for the clusterisation.
    The cartesian positions of each atom over the trajectory are extracted; we obtain a data with high dimensions,
    informing the number of frame, number of selected atoms, xyz positions.
    To efficiently explore the data, its dimension is flatten (number of frame, number of selected atoms * xyz positions).
    INPUTS  :
    - pdb   : absolute path to the pdb file
    - traj  : absolute path to the trajectory file
    - features : atoms selection, use mdAnalysis sytaxes (eg : "protein and name CA")
    - return_features ("protein", default) : selection of the features for the trajectories of the clustered structures
    OUTPUTS :
    - u : mdAnalysis Universe
    - features_xtc  : atoms selection of the whole protein(s). It will be used later when  creating trajectory files of 
    each cluster.
    - features_flat : Set of features used for the clusterization 
    """
    # --- Selection of the features
    u = mda.Universe(pdb, traj)
    features_to_clust = u.select_atoms(features)
    features_xtc      = u.select_atoms(return_features) # When generating the cluster.xtc
    
    # Subsample data if too large # default is False, no subsampling
    index_subsample = subsample(u, percentage_subsample)

    # Flatten the features
    features_flat = []
    for i, frame in enumerate(u.trajectory[index_subsample]) :
        features_flat.append(np.concatenate(features_to_clust.positions))

    return(u, features_xtc, features_flat)

def dim_reduc(features_flat) :
    """ --- Run dimensionality reduction ---
    This function rescale/normalize the data.
    PCA (Principal Component Analysis) is performed to generate a data with lower dimension.
    A number of dimensions (components) that explain above 80% of the data variance is chosen.
    INPUT  :
    - features_flat : Set of features used for the clusterization 
    OUTPUT :
    - data_pca : data with lower dimension
    - The plot of explained variance and the projection of the data points onto the two first components
    """
    # - Rescale the data
    scaler = StandardScaler()
    X_features = scaler.fit_transform(features_flat)
    
    # - Perform PCA and the explained variance of each component
    # - Transform the data into lower dimension, with nb of components explaining 80% of the data variance
    pca = PCA()
    pca.fit_transform(X_features)
    pca_variance = [ i/sum(pca.explained_variance_) for i in pca.explained_variance_ ] 
    cum_var_exp  = np.cumsum(pca_variance)
    nb_comp      = np.shape( np.where(cum_var_exp <= 0.8) )[1]
    
    pca2     = PCA(n_components = nb_comp)
    data_pca = pca2.fit_transform(X_features)
    
    print( "A dimensionnality reduction was performed on your data, using {} components".format(nb_comp) )
    return data_pca

#def plot_components() :
#    look      = 10
#    labelsize = 16
#    fig, ax   = plt.subplots(1,2, figsize=(10,4), gridspec_kw={'width_ratios': [6,4], 'wspace': 0.3})
#
#    ax[0].bar(range(1,look+1), pca_variance[:look], alpha = 0.5,
#              align ='center', label = 'individual explained variance')
#    ax[0].step(range(1,look+1), cum_var_exp[:look], where='mid',
#             label='cumulative explained variance')
#    ax[0].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
#    ax[0].legend(fontsize = labelsize-3)
#    ax[0].set_ylabel('Variance ratio', fontsize = labelsize)
#    ax[0].set_xlabel('Principal components', fontsize = labelsize)
#
#    ax[1].scatter(data_pca[:,0], data_pca[:,1])
#    ax[1].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
#    ax[1].set_ylabel('PC1', fontsize = labelsize)
#    ax[1].set_xlabel('PC2', fontsize = labelsize)
#    return


def plot_projection(index_den, dist_reach):
    """ Return figures of a reachability plot and the projection of the data with lower dimension """
    # - This funtion is executed when the function transform_reach is called 
    labelsize = 13
    ax[1].plot( range(len(index_den)), dist_reach )
    ax[1].set_ylabel("Reachability \n distance", fontsize = labelsize)
    ax[1].set_ylabel("Data points", fontsize = labelsize)
    ax[1].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
    ax[1].set_title("Reachability plot", fontsize = labelsize)
    ax[0].set_title("Dendogram / Hierarchichal clustering", fontsize = labelsize)
    ax[0].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
    return

