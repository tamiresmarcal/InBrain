import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting, datasets, surface  

class SummaryVisualization:
    """
    A class to visualize summary data on brain surfaces.

    Attributes:
    fsaverage_mesh_right (str): Path to the right hemisphere mesh file.
    fsaverage_sulc_right (str): Path to the right hemisphere sulcus file.
    fsaverage_annot_right (str): Path to the right hemisphere annotation file.
    fsaverage_mesh_left (str): Path to the left hemisphere mesh file.
    fsaverage_sulc_left (str): Path to the left hemisphere sulcus file.
    fsaverage_annot_left (str): Path to the left hemisphere annotation file.
    economo_summary_path (str): Path to the economo summary CSV file.
    rh_summary_fsaverage (pd.DataFrame): DataFrame containing right hemisphere summary data.
    lh_summary_fsaverage (pd.DataFrame): DataFrame containing left hemisphere summary data.
    description (pd.Index): Column names of the summary DataFrame.
    """

    def __init__(self, paths):
        """
        Initializes the SummaryVisualization class with the given paths.

        Parameters:
        paths (dict): Dictionary containing paths to the necessary files.
        """
        self.fsaverage_mesh_right = paths['fsaverage_mesh_right']
        self.fsaverage_sulc_right = paths['fsaverage_sulc_right']
        self.fsaverage_annot_right = paths['fsaverage_annot_right']
        self.fsaverage_mesh_left = paths['fsaverage_mesh_left']
        self.fsaverage_sulc_left = paths['fsaverage_sulc_left']
        self.fsaverage_annot_left = paths['fsaverage_annot_left']
        self.economo_summary_path = paths['economo_summary_path']
        
        # models summary
        data = pd.read_csv(self.economo_summary_path, index_col = 0)
        rh_data = data[data.Hemisphere == 0]
        lh_data = data[data.Hemisphere == 1]
        # mesh of average brain to create visualization on it
        rh_mesh = pd.Series(list(surface.load_surf_data(self.fsaverage_annot_right)), name='atlas').reset_index()
        lh_mesh = pd.Series(list(surface.load_surf_data(self.fsaverage_annot_left)), name='atlas').reset_index()
        # merge
        self.rh_summary_fsaverage = rh_data.merge(rh_mesh, left_on='atlas_x', right_on='atlas', how='right')
        self.lh_summary_fsaverage = lh_data.merge(lh_mesh, left_on='atlas_x', right_on='atlas', how='right')
        self.description = self.rh_summary_fsaverage.columns 


    def plot_cortex(self, column, cmap, threshold = False):
        """
        Plots the histogram and surface map of a specified column.

        Parameters:
        column (str): Column name to be plotted.
        cmap (str): Colormap to be used for plotting.
        threshold (bool): Whether to apply a threshold to the plot.
        """
        plt.hist(self.lh_summary_fsaverage[[column]], bins=40, color = '#582c9d')
        plt.title("Histogram of "+column+" in left hemisphere")
        plt.show()

        for view in ['lateral', 'medial']:
            plotting.plot_surf_stat_map(surf_mesh = self.fsaverage_mesh_left, 
                                              stat_map = np.array(self.lh_summary_fsaverage[[column]]), 
                                              bg_map = self.fsaverage_sulc_left,
                                              hemi = 'left', 
                                              view = view,
                                              title = column +' | '+view +' view of left hemisphere ',
                                              colorbar = True,
                                              cmap = cmap,
                                              threshold=threshold,
                                              symmetric_cbar = False,
                                              bg_on_data = True,
                                                   ) 
            plotting.show()

            plotting.plot_surf_stat_map(surf_mesh = self.fsaverage_mesh_right, 
                                              stat_map = np.array(self.rh_summary_fsaverage[[column]]), 
                                              bg_map = self.fsaverage_sulc_right,
                                              hemi = 'right', 
                                              view = view,
                                              title = column +' | '+view +' view of right hemisphere ',
                                              colorbar = True,
                                              cmap = cmap,
                                              threshold=threshold,
                                              symmetric_cbar = False,
                                              bg_on_data = True,
                                                   ) 
            plotting.show()


    def plot_atlas_all(self, cmap = 'Purples'):
        """
        Plots all atlas structures on the left hemisphere.

        Parameters:
        cmap (str): Colormap to be used for plotting.
        """
        dummies_left = pd.get_dummies(self.lh_summary_fsaverage.atlas)
        for col in dummies_left.columns:
            name = str(np.array(self.lh_summary_fsaverage[self.lh_summary_fsaverage.atlas == col].structure_name.head(1))[0])
            plotting.plot_surf_stat_map(surf_mesh = self.fsaverage_mesh_left, 
                                              stat_map = np.array(dummies_left[col].astype(int)), 
                                              bg_map = self.fsaverage_sulc_left,
                                              hemi = 'left', 
                                              view = 'lateral',
                                              title = name +' | Structure: '+ str(col),
                                              colorbar = False,
                                              cmap = cmap,
                                              threshold=0.5,
                                              bg_on_data = True,
                                                   ) 
            plotting.show()
                
                
    def plot_atlas(self, n_structure, cmap = 'Purples'):
        """
        Plots a specific atlas structure on the left hemisphere.

        Parameters:
        n_structure (int): Atlas structure number to be plotted.
        cmap (str): Colormap to be used for plotting.
        """
        dummies_left = pd.get_dummies(self.lh_summary_fsaverage.atlas)
        name = str(np.array(self.lh_summary_fsaverage[self.lh_summary_fsaverage.atlas == n_structure].structure_name.head(1))[0])
        for view in ['lateral','medial','dorsal','ventral']:
            plotting.plot_surf_stat_map(surf_mesh = self.fsaverage_mesh_left, 
                                              stat_map = np.array(dummies_left[n_structure].astype(int)), 
                                              bg_map = self.fsaverage_sulc_left,
                                              hemi = 'left', 
                                              view = view,
                                              title = name +' | Structure: '+ str(n_structure)+' | ' +view +' view of left hemisphere ',
                                              colorbar = False,
                                              cmap = cmap,
                                              threshold=0.5,
                                              bg_on_data = True,
                                                   ) 
            plotting.show()
