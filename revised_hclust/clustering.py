#!/usr/bin/env python3.7
# edit
from revised_hclust.processing import plot_projection

import numpy as np
import pandas as pd
import random
from multiprocessing import Pool

import MDAnalysis as mda

from sklearn.decomposition import PCA
#from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

"""
This module compute the clustering of the pre-processed data.
It computes the hierarchical clustering that generate the dendogram,
compute the reachability plot transformation, set the generation of cutoff list which is used to
define (and refine) the clusters.
At the end, the labeling of the clusters are modified, the clusters' size are ranked in descending order.
"""

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
    tag_iteration = ""

    # For the first generation (raw data), I need to add manually, because of delimiter can not be iterated
    visited_parent = np.append(visited_parent, ID )
    used_cutoff    = np.append(used_cutoff, list_cutoff[0])
    used_cutoff_i  = np.append(used_cutoff_i, 0)
    used_delimiter = np.append( used_delimiter, np.array([0,len(dist_reach)]).reshape(1,2), axis=0 ) # delimiter, eg : [0,20000]
    engender_child.append( 0 ) # Before splitting, parent do not have child
    tag_child.append( 0 )

    # --- Splitting data successively according to the value list_cutoff ---
    while cuttof_i < len(list_cutoff)-1 :   
        if seek_ID > 0 and (np.shape(used_delimiter)[0] <= seek_ID) :
            # -- sometimes, unsplitted data (due to their min_number_data) are registered, and it generated error when
            # min_number_data is reached but the cutoff_ iteration is not yet finish
            tag_iteration = "stop"
            return visited_parent, used_cutoff, used_delimiter, engender_child, tag_child, tag_iteration

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
            # If none of the child has >  min_number_data, this means the data can not be splitted anymore
            # send an error message
            children_member_count = any(nb_member > min_number_data for nb_member in np.diff(splitting_delimitation) )
           
            if children_member_count == True :            

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
                tag_iteration = "continue"

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

            tag_iteration = "continue"

        seek_ID += 1
    return visited_parent, used_cutoff, used_delimiter, engender_child, tag_child, tag_iteration


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
    if np.shape(tag_child)[0] ==1 :
        last_tag_child = 0

    elif  np.shape(tag_child)[0] > 1:
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

