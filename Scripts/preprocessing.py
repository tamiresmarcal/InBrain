import os
import shutil
import numpy as np
import pandas as pd
from nilearn import surface

def collector(participants_list, path_example):
    """
    Collects and processes data for a list of participants.

    Parameters:
    participants_list (list): List of participant IDs.
    path_example (str): Example path to the data.

    Returns:
    list: A list containing:
        - participants_list_completed_index (pd.DataFrame): DataFrame of completed participant indices.
        - participants_list_completed (pd.DataFrame): DataFrame of completed participants.
        - participants_list_incompleted (np.ndarray): Array of incomplete participants.
        - df_vertices (pd.DataFrame): DataFrame of vertex data.
        - df_strutures_BN (pd.DataFrame): DataFrame of structural data.
    """
    # participants
    participants_list = os.listdir(freesurfer_data_folder)
    participants_list = np.setdiff1d(participants_list, ['fsaverage'])

    # data format
    df_dict = {"participant":[],"hemisphere":[],"atlasEcono":[],"atlasBN":[],"atlasDKT":[], "area":[],"curv":[],"sulc":[],"thickness":[]}
    df_dict_ = {"atlasEcono":[],"atlasBN":[],"atlasDKT":[], "area":[],"curv":[],"sulc":[],"thickness":[]}
    df_atlasBN = {}
    participants_list_incompleted = []

    for i, participant in enumerate(participants_list):
        print(i)
        for hemisphere in ['lh', 'rh']:

            # paths
            atlasEcono = ["/label/"+str(hemisphere)+".Eco.annot", 'atlasEcono']
            atlasBN = ["/label/"+str(hemisphere)+".BN_Atlas.annot", 'atlasBN']
            atlasDKT = ["/label/"+str(hemisphere)+".aparc.DKTatlas.annot", 'atlasDKT']
            area = ["/surf/"+str(hemisphere)+".area", 'area']
            curv = ["/surf/"+str(hemisphere)+".curv", 'curv']
            sulc = ["/surf/"+str(hemisphere)+".sulc", 'sulc']
            thickness = ["/surf/"+str(hemisphere)+".thickness", 'thickness']
            features = [atlasEcono, atlasBN, atlasDKT, area, curv, sulc, thickness]

            try:
                # vertice data
                for [feature, f] in features:
                    data = surface.load_surf_data(freesurfer_data_folder + participant + feature)
                    df_dict[f].extend(data)
                    df_dict_[f].extend(data)
                df_dict['participant'].extend([participant] * len(data))
                df_dict['hemisphere'].extend([hemisphere] * len(data))

                # strutural data
                df_atlasBN_ = pd.DataFrame(df_dict_).groupby(by='atlasBN').mean().drop(columns=['atlasEcono', 'atlasDKT'])
                df_atlasBN_['participant'] = [participant] * len(df_atlasBN_)
                df_atlasBN_['hemisphere'] = [hemisphere] * len(df_atlasBN_)
                df_atlasBN[participant+hemisphere] = df_atlasBN_
                df_dict_ = {"atlasEcono":[],"atlasBN":[],"atlasDKT":[], "area":[],"curv":[],"sulc":[],"thickness":[]}

            # participants incompleted
            except:
                participants_list_incompleted.append(participant) 
                
    # to dataframes            
    df_strutures_BN = pd.concat(df_atlasBN).droplevel(level=0).reset_index()
    df_vertices = pd.DataFrame(df_dict)
    
    # participants
    participants_list_incompleted = np.unique(participants_list_incompleted)
    participants_list_comp = [[i, sub] for i, sub in enumerate(participants_list) if sub not in participants_list_incompleted]
    participants_list_completed_index = pd.DataFrame(participants_list_comp)[0]
    participants_list_completed = pd.DataFrame(participants_list_comp)[1]
    
    # all info
    all_info = [participants_list_completed_index, 
                participants_list_completed, 
                participants_list_incompleted,
                df_vertices, df_strutures_BN] 
    return all_info
    
    
class SuferData:
    """
    A class to handle and process Freesurfer data.

    Attributes:
    freesurfer_data_folder (str): Path to the Freesurfer data folder.
    path_base_mri (str): Base path for MRI data.
    path_example (str): Example path to the data.
    participants_list_completed_index (pd.DataFrame): DataFrame of completed participant indices.
    participants_list_completed (pd.DataFrame): DataFrame of completed participants.
    participants_list_incompleted (np.ndarray): Array of incomplete participants.
    df_vertices (pd.DataFrame): DataFrame of vertex data.
    df_strutures_BN (pd.DataFrame): DataFrame of structural data.
    """

    def __init__(self,
                 freesurfer_data_folder = "/home/brunovieira/Tamires_Experiments/Data/Nki_data/",
                 path_base_mri = "/home/brunovieira/Tamires_Experiments/Data/",
                 path_example = "/home/brunovieira/Tamires_Experiments/Data/Nki_data/A00008326/"
                    ):
        """
        Initializes the SuferData class with the given paths and processes the data.

        Parameters:
        freesurfer_data_folder (str): Path to the Freesurfer data folder.
        path_base_mri (str): Base path for MRI data.
        path_example (str): Example path to the data.
        """
        participants_list = os.listdir(freesurfer_data_folder)
        participants_list = np.setdiff1d(participants_list, ['fsaverage'])

        all_info = collector(participants_list, path_example) 
        self.participants_list_completed_index = all_info[0]
        self.participants_list_completed = all_info[1]
        self.participants_list_incompleted = all_info[2]
        self.df_vertices = all_info[3] 
        self.df_strutures_BN = all_info[4]

        
    def save_files(self):
        """
        Saves the processed data to parquet files.
        """
        self.df_strutures_BN.to_parquet(self.path_base_mri +"/Brainnetome/data_894_BN.parquet")
        self.df_vertices.to_parquet(self.path_base_mri + "MRI_Data_Vertices_" + str(len(self.df_vertices.participant.unique())) + ".parquet")

        
    def monitor(self):
        """
        Prints the number of completed and incomplete participants.
        """
        print('Completed: '+str(len(self.participants_list_completed)))
        print('Incompleted: '+str(len(self.participants_list_incompleted)))
