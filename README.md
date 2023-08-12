# Revised hierarchical clustering (RHCC)
## Introduction
The Revised Hierarchical Clustering is an automated algorithm to select clusters from a hierarchical representation, by transforming this latter into a reachability plot.
Thus, the RHCC merge the concept of the hierarchical clustering algorithm and density based clustering OPTCS.
The general idea was inspired from the paper work of Sander and collaborators : DOI:10.1007/3-540-36175-8_8.
The motivation behind the development of RHCC emerged from the difficulty of reading a dendogram, and defining optimum hyperparameters of OPTIC for a large data set.

The basic hierarchical clustering algorithm (HCC) computes the distance between two groups and merge them according to their similarity until forming one large group.
Thus, the algorithm yield to a hierarchical representation of the data points, called dendogram.
The HCC does not define explicitely the final (optimum) clusters of the data -- the users have to define a horizontal cutoff through the dendogram.
This horizontal cutoff is defined by visual inspection.

OPTICS is a density-based algorithm that computes the shortest distance (reachability distance) "walk" from one point to its neighbor. Two parameters need to be defines, $\epsilon$ and Minpoints.
For a very large data, the time execution of the algorithm increase significantly.
The projection of the reachability distance results to a reachability plot where potential clusters are indicated by "dents". 

Protocols of the RHCC :<br>
(1) The RHCC computes a hierarchical clustering, which yield into a dendogram. <br>
The dendogram height between two singleton cluster (from the left to the the right) are stored as it corresponds to the reachability distance.<br>
(2) The projection of the reachability distances give a reachability plot.<br>
(3) From the tops to the bottoms of the tree, at separable dents, clusters are splitted into finer child(ren) cluster(s), i.e into more homogenous clusters.<br>
![protocol | width=50](images/reachability_plot_0.png )
<img src="images/reachability_plot_0.png" width="150" height="280">


## Installation
Please dowload the file **revised_huclust.py*** into a directory
Add the following command to your .bashrc : <br>
`export PYTHONPATH=$PYTHONPATH:/path_to_directory/`

## Usage
This module is executable in IDE such as Jupyter-notebook.
After installation, the module can be import into Jupyter-notebook : <br>
`import revised_hclust as r_clust`

To perform the clustering, execute the command :
`r_clust.execute_revised_hclust(pdb, traj, feat, cutoff_min, min_number_data, outcomb)`

where : <br>
- pdb : absolute path to the pdb file
- traj : absolute path to the trajectory file
- features : atoms selection, use mdAnalysis sytaxes (eg : "protein and name CA")
- cutoff_min (int, float) :  define maximal distance value between two points to be considered as similar.
- min_number_data : define the minimum number of points to be considered as clusters
- outcomb : absolute path where to write the trajectory files for the clustered points

Thus, the function execute_revised_hclust compute the RHCC, generate trajectory files of the clustered structures, and display several analysis plots.





