#!/usr/bin/env python3.7
# edit

from revised_hclust.processing import features_selection, dim_reduc
from revised_hclust.clustering import transform_reach, define_cutoff, execute_clustering, plot_reachability, label_clustering, boxplot_

import numpy as np
import pandas as pd
import random
from multiprocessing import Pool

import MDAnalysis as mda

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

import time

"""
This module use deep learning techniques to optmize the clustering.
This module applies dropout and the computation of loss of function (sum squared error) to optmize the variable cutoff. 
This cutoff is used to split data into finer clusters.
Several function were written to falicitate the implementation (iteration) of the optimization.
"""

def rescale_data(data_pca):
    r_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(data_pca)
    return r_data

def goal_prediction(data_pca, label_) :
    # data_pca is the raw data, it is fixed
    goal_pred = []
    for label_i in np.unique(label_) :
        index_label = np.where(label_ == label_i)
        goal_pred_i = np.mean(data_pca[index_label], axis=0)
        goal_pred.append(goal_pred_i)
    return np.array(goal_pred)

def dropout(label_) :
    # --- dropout is used to avoid overfitting, especially in large data
    mask = 30
    len_data = np.shape(label_)[1]
    dropout_mask = np.ones(len_data,dtype=int)
    mask_index = np.random.choice( np.arange(len_data), size = int((len_data*mask)/100) )
    dropout_mask[mask_index] = 0
    return dropout_mask.reshape(dropout_mask.shape[0],1)

def gradient_sse(data_pca, label_, weight, mask=30) :
    # data_pca variate here, it can be multiplied by the weight. It represent the predicted values according to the optimization of the weight
    # Stochastic gradient : the weight will be updated by passing from each cluster
    # Reciprocal is the factor to compensate for the removal of a certain amount of data
    """
    abs_diff = is the absolute difference (error)
    sse = sum squarred error, within a cluster is computed the distance of each point to the centroids.
    The aim is to minimize the sse
    INPUTS : 
    - data_pca : data with a dimension (m,n)
    - label : list of cluster's label
    OUTPUT :
    sse : sum squared error
    """

    lr  = 0.001
    sse = 0
    reciprocal_factor = 1 / (1-(mask/100))

    r_data    = rescale_data(data_pca)
    goal_pred = goal_prediction(r_data, label_)
    for enum_i, label_i in enumerate(np.unique(label_)) :
        index_label = np.where(label_ == label_i)

        dropout_mask  = dropout(index_label)
        reduced_data  = r_data[index_label] * dropout_mask * reciprocal_factor
        weighted_data = reduced_data * weight
        
        # - Compute sum squared error
        error = np.sum( (weighted_data - goal_pred[enum_i])**2)
        sse  += error

        # - Update the weight
        gradient = np.sqrt( np.sum(weighted_data - goal_pred[enum_i]) )
        weight   = weight - (lr*gradient)

    return weight, sse


def generate_xtc(u, features_xtc, index_den, label_, outcomb) :
    """ --- Generate trajectory files for each cluster --- 
    This function will use mdAnalysis to create the trajectory files
    INPUTS :
    - u : mdAnalysis Universe
    - features_xtc : select atoms to view in the clustering, use mdAnalysis  (default : "protein")
    - index_den    : points index, arranged accordingly to hierarchical clustering
    - label        : list of label assignement after clusterization
    - outcomb      : path to write the trajectory files 
    OUTPUTS :
    - trajectory files are writen in the directory "outcomb"
    """
    for enum_i, real_clust_id in enumerate(np.unique(label_)[:-1]) :
        data_clust_index = index_den[ np.where(label_==real_clust_id) ]

        open(outcomb + "clust_"+ str(enum_i) +".cat.xtc", 'a').close()
        with mda.Writer(outcomb + "clust_"+ str(enum_i) +".cat.xtc") as W:
            for frame in u.trajectory[data_clust_index] :
                W.write(features_xtc)
    return

# -- Load data and transform it.
def process_data(pdb, traj, features, method='ward', return_features="protein", percentage_subsample=False ) :
    u, features_xtc, features_flat = features_selection(pdb, traj, features, return_features=return_features, percentage_subsample=percentage_subsample)
    data_pca = dim_reduc(features_flat)
    index_den, dist_reach = transform_reach(data_pca, method=method)
    return u, index_den, data_pca, dist_reach
 
def perform_rhc(dist_reach, min_number_data, cutoff_min,
        return_plot_reachability=False, return_boxplot=False) :
    interval_, list_cutoff = define_cutoff(dist_reach, cutoff_min)
    visited_parent, used_cutoff,  used_delimiter, engender_child, tag_child, tag_iteration = execute_clustering(min_number_data, list_cutoff, dist_reach)
    label_ = label_clustering(dist_reach, visited_parent, used_delimiter, tag_child)

    if return_plot_reachability==True:
        plot_reachability(dist_reach, interval_,  visited_parent, used_cutoff, used_delimiter, engender_child, tag_child)
    if return_boxplot==True:
        boxplot_(label_, dist_reach)
    return label_, tag_iteration

def single_rhc(pdb, traj, features, cutoff_min, min_number_data, outcomb, method = 'ward', return_features="protein", percentage_subsample=False):
    data_pca, dist_reach = process_data(pdb, traj, features, method = method, return_features=return_features, percentage_subsample=percentage_subsample)
    perform_rhc(dist_reach, min_number_data, cutoff_min)
    return

def deep_rhcc(pdb, traj, features, min_number_data, outcomb, cutoff_min=None , iteration=50, method='ward',
        return_features="protein", return_plot_reachability=True, return_boxplot=False, return_xtc_file=False, show_steps=True, percentage_subsample=False ):
    """
    The function deep_rhcc optimize the cutoff_min and computed the clusterization.
    INPUTS :
    ------
    - pdb  : path to pdb 
    - traj : path to trajectory file
    - min_number_data : define the minimum number of points to be considered as clusters
    - outcomb : path to write the trajectory files
    - cutoff_min (None, default) : By default, a value of cutoff_min is initiated randaomly, 
        Users can also add a fix value (int or float)
    - iteration  (50, default) : Optimization of cutoff_min stops when the clusterinzation can not be refine anymore.
        Users can modify the number of iteration (int)
    - method : {'ward', 'single', 'complete', 'average'}, default='ward' : this is a linkage method to compute distance between clusters
        Other options : 'single', 'complete'
    - return_plot_reachability (True, default) : set False, if you do not want to display the reachability and the cutoff_min refinement
    - return_boxplot  (False, default) : set True, if you would like to display boxplot and analysis of data frequency
    - return_xtc_file (False, default) : set True, to generate xtc files for each clusters
    - show_steps (True, default) : Display iteration steps, the sum squared error and the optimized cutoff_min distance
    - percentage_subsample : {False or float} (False, default) 
        As default, no subsampling is executed.
        If a float value (between 1 and 100), the percentage of data to keep after a random subsampling.
        For enough large data, it is recommended to subsample data, it will not modify the
        consistency of the data and it will improve the efficiency of the algorithm.

    OUTPUTS :
    -------
    - index_den : real index of the data shuffled after clusterization (generated from the dendogram)
    - label_    : labelistation of each data points, following the indexation from index_den
    """

    # I want to optimize the cutoff_min, 
    # In ML, this cutoff_min will correspond to the weight. This weight will be optimize

    start_time = time.time()
    # --- Load data and transform data
    _, index_den, data_pca, dist_reach = process_data(pdb, traj, features, method=method, return_features=return_features, percentage_subsample=percentage_subsample)

    # --- Initializing cutoff ---
    # If cutoff_min == None (default), no value was given, then randomly choose a cutoff_min
    if cutoff_min is None :
        interval__, _ = define_cutoff(dist_reach, np.sort(dist_reach)[::-1][10] )
        cutoff_min    = random.uniform(interval__+1,  np.sort(dist_reach)[::-1][5] ) #  np.max(dist_reach)-(interval__+1))
        print("Initialization of the cutoff at {} {}".format(cutoff_min, interval__))
    
    weight_cutoff = cutoff_min
    label_ = 0

    # --- Perform clustring and iterate
    for it in range(iteration) :
        label_, tag_iteration = perform_rhc( dist_reach, min_number_data, weight_cutoff)

        # Continue to iterate if I do not loose X% of my data and I did not create a negative cutoff
        nb_outliers  = np.shape( np.where(label_ == 999))[1]
        max_outliers = (data_pca.shape[0]*36)/100
        if (nb_outliers >= max_outliers) or ( weight_cutoff < 0) :
            break

        if tag_iteration == "stop" :
            print("!!!!! Reconsider to reduce the value corresponding to min_number_data !!!!!")
            break
        
        weight, sse = gradient_sse(data_pca, label_, weight_cutoff)
        weight_cutoff = weight
        if show_steps==True :
            print("iter : {} ---- The squared error is {:.2f}:  --- Cutoff {:.2f}".format(it, sse, weight_cutoff))
  
    # --- Generate trajectory files for each cluster ---

    if return_xtc_file==True :
        u, index_den, _, _ = process_data(pdb, traj, features, method=method, return_features=return_features, percentage_subsample=percentage_subsample )
        features_xtc       = u.select_atoms(return_features)
        generate_xtc(u, features_xtc, index_den, label_, outcomb)

    if return_plot_reachability==True :
        perform_rhc(dist_reach, min_number_data, weight_cutoff, return_plot_reachability=True) 
        print("The cutoff distance is : {}".format(weight_cutoff))

    if return_boxplot== True :
        perform_rhc(dist_reach, min_number_data, weight_cutoff, return_boxplot=True) 

    end_time = time.time()
    print("Time of script execution ", (end_time - start_time)/60) 

    return index_den, label_ 


