*In the near future, I would like to integrate this script in deep learning, to choose wisely all the input hyper-parameters

# Revised hierarchical clustering (RHCC)
## Introduction
The Revised Hierarchical Clustering (RHCC) aims to cluster structures of biomolecules sharing similar conformations. <br>
The RHCC is an automated algorithm that defines clusters from a hierarchical representation, and computes a transformation of this latter into a reachability plot.
Thus, the RHCC merge the concept of the hierarchical clustering algorithm and density based clustering OPTCS.
The general idea was inspired from the paper work of Sander and collaborators : DOI:10.1007/3-540-36175-8_8.
The motivation behind the development of RHCC emerged from the difficulty of reading a dendogram, and defining optimum hyperparameters of OPTIC with large data set.

The basic hierarchical clustering algorithm (HCC) computes the distance between two groups and merge them according to their similarity until forming a single large group.
Thus, the algorithm yield to a hierarchical representation of the data points known as a dendogram.
The HCC does not define explicitely the final (optimum) clusters of the data -- the users have to define a horizontal cutoff through the dendogram.
This horizontal cutoff is defined by visual inspection.

OPTICS is a density-based algorithm that computes the shortest distance (reachability distance) "walk" from one point to its neighbor. It requires two parameters, $\epsilon$ and Minpoints.
For a very large data, the time execution of the algorithm increase significantly.
The projection of the reachability distance results to a reachability plot where potential clusters are indicated by "dents". 

Protocols of the RHCC :<br>
(1) The RHCC computes a hierarchical clustering, which yield into a dendogram. <br>
The dendogram heights between two singleton cluster (from the left to the the right) are stored as they correspond to the reachability distance.<br>
(2) The projection of the reachability distances give a reachability plot.<br>
(3) From the top to the bottom of the dendogram, at separable dents, clusters are splitted into finer child(ren) cluster(s), i.e into more homogenous clusters.<br>
<img src="images/reachability_plot_0.png" width="500" >


## Installation
Please dowload the file **revised_huclust.py** into a directory.
Add the following command to your .bashrc : <br>
`export PYTHONPATH=$PYTHONPATH:/path_to_directory/`

## Usage
This module can be executed in IDEs such as Jupyter Notebook. After installation, import the module into Jupyter Notebook: <br>
`import revised_hclust as r_clust`

To perform the clustering and generate trajecories files of clusters at once, execute the command :
`r_clust.execute_revised_hclust(pdb, traj, feat, cutoff_min, min_number_data, outcomb)`

where : <br>
- pdb : absolute path to the PDB file
- traj : absolute path to the trajectory file
- features : atoms selection using mdAnalysis syntax (e.g. : "protein and name CA")
- cutoff_min (int, float) :  define maximal distance value between two points to be considered similar
- min_number_data : define the minimum number of points to be considered as clusters
- outcomb : absolute path to write the trajectory files for the clustered points <br>
e.g. : <br>
```
pdb  = "/path/to/directory/file.pdb"
traj = "/path/to/directory/file.xtc"
features   = "protein and name CA"
cutoff_min = 4
min_number_data = 400 
outcomb = "/path/to/directory/" 
r_clust.execute_revised_hclust(pdb, traj, feat, cutoff_min, min_number_data, outcomb)
```

Thus, the function execute_revised_hclust compute the RHCC, generate trajectory files for the clustered structures, and display several analysis plots.


Alternatively, to analyze each step of the module, execute the following functions in order: <br>
```
r_clust.features_selection()
   r_clust.dim_reduc()
   r_clust.transform_reach()
   r_clust.define_cutoff()
   r_clust.execute_clustering()
   r_clust.plot_reachability()
   r_clust.label_clustering()
   r_clust.boxplot_()
   r_clust.generate_xtc()
```

To display the description, input(s) and output(s) of each function, execute `r_clust.features_selection?` or `r_clust.dim_reduc?`, etc.
















