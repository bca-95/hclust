#!/usr/bin/env python3.7
# edit
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

def compute_linkage(piece_data, method='ward') :
    # I call it piece_data, as I will compute to multi-process that will divise the data in multiple piece
    distance_matrix = pdist(piece_data, metric='euclidean')
    hclust_matrix   = sch.linkage(distance_matrix, method = method)
    return hclust_matrix

def calculate_linkage_multiprocess(data_pca, num_processes=4):
    piece_data = np.array_split(data_pca, num_processes)
    pool = Pool(processes=num_processes)
    linkage_matrices = pool.map(compute_linkage, piece_data)
    pool.close()
    pool.join()

    # Renumber clusters of the splitted data and concatenate the results
    final_linkage_matrix = linkage_matrices[0]
    for i in range(1, num_processes):
        num_clusters_previous = final_linkage_matrix.shape[0] 
        linkage_matrices[i][:, [0, 1,3]] += num_clusters_previous
        final_linkage_matrix  = np.concatenate((final_linkage_matrix, linkage_matrices[i]), axis=0)

    return final_linkage_matrix


def transform_reach(data_pca, method='ward',num_processes=4, return_projection=False) :
    """ --- This function perform a hierarchical clustering, transform this latter in reachability distance plot  ---
    For more information about reachability distance plot, read : DOI:10.1007/3-540-36175-8_8
    INPUT   : 
    - data_pca   : data with a dimension (m,n)
    OUTPUTS :
    - index_den  : index list of the data points, with an order corresponding from the left to the right of the x-axis 
    of the dendogram
    - dist_reach : list of reachable distances
    - The plot of the dendogram/hierarchical clustering and the plot of the reachability plot are displayed.
    """
    #linkage_  = calculate_linkage_multiprocess(data_pca, num_processes=num_processes)
    linkage_   = sch.linkage(data_pca, method=method, metric="euclidean")
    den_complete = sch.dendrogram(linkage_, no_plot=True) 
    index_den  = np.array(den_complete['leaves'])
    dist_reach = np.linalg.norm( data_pca[ index_den[:-1] ] - data_pca[ index_den[1:] ] , axis=1 )
    dist_reach = np.insert(dist_reach, 0, 0)

    if return_projection==True :
        plot_projection(index_den, dist_reach)

    return index_den, dist_reach


def define_cutoff(dist_reach, cutoff_min) :
    """ --- This function defines a list of discretized cutoff, in descending order. ---
    This list will be used in the function "execute_clustering".
    INPUTS  :
    - cutoff_min : maximal distance value between two points to be considered as similar.
    - dist_reach : list of reachable distances of the whole data
    OUTPUTS :
    - interval_   : interval distance used to create the list of discretized cutoff
    - list_cutoff : list of discretized cutoff
    """
    # Sort the cutoff to choose discritized value
    # Select distance values above the cutoff_min
    # Use the std of the whole selected distance values, as interval to set a list of discretized cutoff 
    sort_cutof  = np.sort(dist_reach)[::-1] 
    
    sort_cutof  = sort_cutof[ sort_cutof >= cutoff_min ]
    interval_   = np.std( np.abs(np.diff(sort_cutof)) )
    list_cutoff = np.arange(cutoff_min, np.max(sort_cutof), interval_)[::-1]
    return interval_, list_cutoff



def create_delimiter(dist_reach, seek_ID, cuttof_i, used_delimiter, list_cutoff) :
    """ --- This function set the delimitation (index of the starting and ending) of each 
    new generated cluster ---
    This function is not meant to be executed idenpendetly, it will be called by the 
    function "execute_clustering" 
    INPUTS :
    - seek_ID  : index of the parent
    - cuttof_i : iterated values from the list of descritized cutoff, generated in the 
    function "define_cutoff"
    - used_delimiter : delimiter index of the parent (start index, end index)
    OUTPUTS :
    - splitting delimitation : index of region to be splitted, in order to generate child(ren)
    - nb_children : number of generated children
    """
    # - Creation of child(ren) :  insert into the parent delimiter, the index region that can be splitted in child(ren)
    # where the reachabilibity distance between two points is above the cutoff
    delimiter_min_max = used_delimiter[seek_ID]
    where_to_split    = np.where(dist_reach[ delimiter_min_max[0]:delimiter_min_max[1] ] > list_cutoff[cuttof_i] )[0]

    splitting_delimitation = 9999
    nb_children = 9999
    # Have more than 1 child
    # Sometimes it re-select the "min" of delimiter_min_max, as it has high distance reachability
    # Remove it
    # Be careful when inserting splitting region, it refers to the index of an already splitted data
    if len(where_to_split)    > 1 :
        if where_to_split[0] == 0 :
            where_to_split    = np.delete(where_to_split, 0)
            where_to_split    = np.arange(delimiter_min_max[0], delimiter_min_max[1])[where_to_split]
            delimiter_min_max[0]   = delimiter_min_max[0]+1
            splitting_delimitation = np.insert(delimiter_min_max, 1, where_to_split) # eg : [0,4,8,20], 4 and 8 were inserted
            nb_children            = np.shape(where_to_split)[0] + 1

        elif where_to_split[0] != 0 :
            where_to_split      = np.arange(delimiter_min_max[0], delimiter_min_max[1])[where_to_split]
            splitting_delimitation = np.insert(delimiter_min_max, 1, where_to_split)
            nb_children            = np.shape(where_to_split)[0] + 1
    
    # Parent might engender 1 fake child(if), or 1 real child (elif) 
    elif len(where_to_split) == 1 :
        if where_to_split[0] == 0 :
            delimiter_min_max[0] = delimiter_min_max[0]+1
            splitting_delimitation = delimiter_min_max
            nb_children            = 1
            
        elif where_to_split[0] != 0 :
            where_to_split    = np.arange(delimiter_min_max[0], delimiter_min_max[1])[where_to_split]
            splitting_delimitation = np.insert(delimiter_min_max, 1, where_to_split)
            nb_children            = np.shape(where_to_split)[0] + 1 #it is always = 2
            
    # No child engendered
    elif len(where_to_split) == 0 :
        splitting_delimitation = delimiter_min_max
        nb_children            = 1

    return splitting_delimitation, nb_children


def execute_clustering(min_number_data, list_cutoff, dist_reach) :
    """ --- This function will performed the clusterization --- 
    The values in the cutoff list will be applied iteratively on the reachability plot.
    While the cutoff_min (maximal value to define points of same clusters) and/or the min_number_data
    is not yet reached, this function continue to dig deep into the reachability plot.
    INPUTS :
    - min_number_data : minimum number of points to be considered as cluster
    - list_cutoff : list of discretized cutoff generated with the function "define_cutoff"
    - dist_reach  : list of reachability distance of the whole data
    OUTPUTS :
    - visited_parent : the parent ID
    - used_cutoff : the applied cutoff to generate children
    - used_delimiter : parent index delimiter
    - engender_child : ID of the engendered child(ren)
    - tag_child      : tag 0 if parent do not have child, else tag 1
    """
    # --- Initializing the clustering ---
    ID       = 0        # ID of the parent
    seek_ID  = 0   # index to seek the parent, analyze if it can have child(ren)
    cuttof_i = 0
    min_number_data = min_number_data
    visited_parent  = np.array([], dtype = int)
    used_cutoff     = np.array([])
    used_cutoff_i   = np.array([], dtype = int)
    used_delimiter  = np.empty((0,2), dtype = int)
    engender_child  = []
    tag_child       = [] # tag 1 if parent have >= 1 child, else tag 0

    # For the first generation (raw data), I need to add manually, because of delimiter can not be iterated
    visited_parent = np.append(visited_parent, ID )
    used_cutoff    = np.append(used_cutoff, list_cutoff[0])
    used_cutoff_i  = np.append(used_cutoff_i, 0)
    used_delimiter = np.append( used_delimiter, np.array([0,len(dist_reach)]).reshape(1,2), axis=0 ) # delimiter, eg : [0,20000]
    engender_child.append( 0 ) # Before splitting, parent do not have child
    tag_child.append( 0 )

    # --- Splitting data successively according to the value list_cutoff ---
    while cuttof_i < len(list_cutoff)-1 :    
        splitting_delimitation, nb_children = create_delimiter(dist_reach, seek_ID, cuttof_i, used_delimiter, list_cutoff)

        # Iterate the cutoff to be used for the following splitting
        if visited_parent[seek_ID] in visited_parent :
            where_i  = np.where(visited_parent == visited_parent[seek_ID])
            cuttof_i = used_cutoff_i[where_i][-1] + 1    
        elif visited_parent[seek_ID] not in visited_parent :
            cuttof_i = cuttof_i +1 

        # Parent have more than one child
        if nb_children    > 1 :
            id_child_list = []        
            for child_i in range(nb_children) :
                child_data_delimiter = [ splitting_delimitation[child_i], splitting_delimitation[child_i+1] ]

                # If the size of data are above X, continue the splitting
                if np.diff(child_data_delimiter) >= min_number_data :
                    # Create a new ID, as it will become a future parent
                    ID += 1
                    id_child_list.append( ID )

                    # Relate the children ID to the parent information, after splitting
                    engender_child[seek_ID] = id_child_list
                    tag_child[seek_ID]      = 1

                    # Preparation of the future parent
                    visited_parent = np.append(visited_parent, ID )
                    used_cutoff    = np.append(used_cutoff, list_cutoff[cuttof_i])
                    used_delimiter = np.append( used_delimiter, np.array(child_data_delimiter).reshape(1,2), axis=0 )
                    used_cutoff_i  = np.append(used_cutoff_i, cuttof_i)

                    # Initiate to 0 the engender child of the futur parent, splitting has not yet been performed
                    engender_child.append( 0) 
                    tag_child.append( 0)

        # If the current parent did engender 0 child (here equal to 1 child)
        # Or engendered only 1 child (its optimized version)
        elif nb_children   == 1 :
            visited_parent = np.append(visited_parent, visited_parent[seek_ID] )
            used_cutoff    = np.append(used_cutoff, list_cutoff[cuttof_i])
            used_delimiter = np.append( used_delimiter, np.array(splitting_delimitation).reshape(1,2), axis=0 )
            used_cutoff_i  = np.append(used_cutoff_i, cuttof_i)
            # After splitting of the parent, child remain 0
            engender_child[seek_ID] = 0 # of the parent after splitting
            tag_child[seek_ID]      = 0
            # Before splitting, we initiate to 0 the child of the futur parent
            engender_child.append(0) # of the futur parent, no splitting yet
            tag_child.append(0)        
        seek_ID += 1
    return visited_parent, used_cutoff, used_delimiter, engender_child, tag_child


def plot_reachability(dist_reach, interval_,  visited_parent, used_cutoff, used_delimiter, engender_child, tag_child) :
    """ --- Plot the reachability plot and draw the cutoff  --- 
    INPUTS :
    - dist_reach
    - interval_ : used to generate discretized list of cutoff
    - visited_parent : parent ID
    - used_cutoff
    - used_delimiter
    - engender_child
    - tag_child
    OUTPUTS : 
    - Plot
    """
    labelsize = 16
    fig, ax = plt.subplots(2,1, figsize=(10,6), gridspec_kw={'height_ratios': [4,1]})
    ax[0].plot( range(len(dist_reach)), dist_reach )
    ax[0].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
    ax[0].set_ylabel("Reachability \n distance", fontsize = labelsize )
    ax[1].plot( range(len(dist_reach)), dist_reach, alpha = 0 )
    ax[1].tick_params(axis='both',which='both',labelbottom=False,bottom=False,labelleft=False, left=False)
    ax[1].spines[['right','left', 'top', 'bottom']].set_visible(False)

    # The index from np.unique will return the index of the first apparition of a pattern
    # original_parent refers to the 1st apparition of the parent ID
    index_parent_with_child = np.where(np.array(tag_child)==1)[0]
    _, original_parent , _  = np.unique(visited_parent, return_index =  True, return_counts = True)
    seek_tagged_child       = 1 # -counter-, 0 is the raw data, before any splitting
    pos_y = -(interval_)  # to draw the cutoff
    
    for i_P in index_parent_with_child :
        count_child = 0
        while count_child != np.shape(engender_child[i_P])[0] :
            # Index to seek delimiter 
            index_delimiter = original_parent[seek_tagged_child]
            width  = used_delimiter[index_delimiter][1] - used_delimiter[index_delimiter][0]
            width  = width - (width*5/100) # Remove 2% of the length for better vizualisation
            # --- Draw the cutoff within the hierachichal representation
            ax[0].add_artist(Rectangle(
                                        xy     = (used_delimiter[index_delimiter][0], used_cutoff[i_P]), \
                                        width  = width, \
                                        height = 0.2, color = "black"
                            )) 
            # --- Draw the cutoff below the representation
            ax[1].add_patch(Rectangle(
                                xy     = (used_delimiter[index_delimiter][0], used_cutoff[i_P]), \
                                width  = width, \
                                height = 0.2, color = "orange", clip_on=False
                            )) 
            count_child += 1
            seek_tagged_child += 1
        pos_y -= interval_
    return


def sort_clusters(label_) :
    # Sort labeling according to the number of element within clusters
    sort_label_ = np.array( [999]*len(label_) )
    unique_label_, nb_elements = np.unique(label_ , return_counts = True)
    sort_label_ele = unique_label_[np.argsort(nb_elements)[::-1]]
    for new_lab, lab in enumerate(sort_label_ele [sort_label_ele != 999]) :
        sort_label_[np.where(label_ == lab)] = new_lab
    return sort_label_


def label_clustering(dist_reach, visited_parent, used_delimiter, tag_child) :
    """ --- Add label to the data, corresponding to the clusters --- 
    INPUTS :
    - dist_reach :
    - visited_parent : Attribution of parent ID of the whole data
    - used_delimiter : Parent index delimiter
    - tag_child : tag 0 if parent did not engender child, else tag 1
    OUTPUT :
    label_ : list of label, 999 refers to the outliers
    """
    # -- Find the last generation
    # Invert the tag_child (invert the list), so we can backpropagate where is located the 
    # last generation (the last individual with kid) before extinction. It corresponds to 
    # the first individual in the list 
    last_tag_child        = np.where( np.array(tag_child)[::-1] == 1 )[0][0]
    invert_visited_parent = visited_parent[::-1]
    last_generation       = invert_visited_parent[:last_tag_child]
    last_generation       = last_generation[ last_generation.argsort() ]

    label_ = np.array([999]*len(dist_reach ))
    clust_ID, index_clust, nb_elements = np.unique(visited_parent, return_index = True, return_counts=True)
    
    for clust_i in last_generation :
        find_delimiter = index_clust[np.where(clust_ID == clust_i)]
        begin_clust    = used_delimiter[find_delimiter][0][0]
        end_clust      = used_delimiter[find_delimiter][0][1]
        label_[begin_clust:end_clust] = clust_i
    # Sort labeling according to the number of element within clusters
    sort_label_ = sort_clusters(label_)
    
    return sort_label_

def boxplot_(label_, dist_reach) :
    """ --- Analyse the homogeneity of the data --- 
    INPUTS : 
    - label_ : list of cluster's label
    - dist_reach
    OUTPUT :
    - boxplot
    """
    df_dist_reach = pd.DataFrame({ 'clust_ID' : label_, 'dist_reach' : dist_reach })

    labelsize = 16
    fig, ax   = plt.subplots(2,1,figsize=(10,8), gridspec_kw={'height_ratios': [5,2], 'hspace': 0.3})
    # --- Boxplot --
    sns.boxplot(x="clust_ID", y="dist_reach",
                        data=df_dist_reach[df_dist_reach['clust_ID'] != 999] , color = "orange",  zorder = 1, ax = ax[0])
    ax[0].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
    ax[0].set_ylabel( "Reachability \n distance",fontsize = labelsize  )
    ax[0].set_xlabel( "Cluster ID",fontsize = labelsize  )

    old_id_clust = np.unique(label_)[:-1]
    new_id_clust = np.arange(0, len(old_id_clust))
    # Set number of ticks for x-axis
    ax[0].set_xticks(new_id_clust)
    # Set ticks labels for x-axis
    ax[0].set_xticklabels(np.arange(1, len(new_id_clust)+1))

    # --- Frequency ---
    label_count = np.unique(label_, return_counts = True)
    frequency_cluster = [(i*100)/np.sum(label_count[1][:-1]) for i in label_count[1][:-1]]
    sns.barplot(label_count[0][:-1], frequency_cluster, color = "gray" )
    ax[1].tick_params (axis = 'both', which = 'major', labelsize = labelsize )
    ax[1].set_ylabel( "Frequency %",fontsize = labelsize  )
    ax[1].set_xlabel( "Cluster ID",fontsize = labelsize  )
    ax[1].set_xticks(new_id_clust)
    ax[1].set_xticklabels(np.arange(1, len(new_id_clust)+1))
    return


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
    sse : sum squared errot
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
    visited_parent, used_cutoff,  used_delimiter, engender_child, tag_child = execute_clustering(min_number_data, list_cutoff, dist_reach)
    label_ = label_clustering(dist_reach, visited_parent, used_delimiter, tag_child)

    if return_plot_reachability==True:
        plot_reachability(dist_reach, interval_,  visited_parent, used_cutoff, used_delimiter, engender_child, tag_child)
    if return_boxplot==True:
        boxplot_(label_, dist_reach)
    return label_

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
        label_ = perform_rhc( dist_reach, min_number_data, weight_cutoff)
        
        # Continue to iterate if I do not loose X% of my data and I did not create a negative cutoff
        nb_outliers  = np.shape( np.where(label_ == 999))[1]
        max_outliers = (data_pca.shape[0]*36)/100
        if (nb_outliers >= max_outliers) or ( weight_cutoff < 0) :
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









