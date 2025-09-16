from boruta import BorutaPy
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import shap
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


def _split_train_test(df, features, structure):
    """
    Split the dataset into training and testing sets based on the structure.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    features (list): List of feature columns.
    structure (int): Structure identifier.

    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    test_index = df[(df.h_structure == '1' + str(structure)) | (df.h_structure == '0' + str(structure))].index
    train_index = set(df.index) - set(test_index)
    X, y = df[features], df.anual_rate
    X_train = np.array(X.loc[train_index])
    X_test = np.array(X.loc[test_index])
    y_train = np.array(y.loc[train_index])
    y_test = np.array(y.loc[test_index])
    Xy = X_train, X_test, y_train, y_test
    return Xy


def _model_train(X_train, y_train):
    """
    Train a LightGBM model.

    Parameters:
    X_train (np.array): Training features.
    y_train (np.array): Training target.

    Returns:
    LGBMRegressor: Trained LightGBM model.
    """
    lgbm = LGBMRegressor()
    lgbm.fit(X_train, y_train)
    return lgbm


def _model_test(lgbm, X_test, y_test):
    """
    Test the trained model and calculate R2 score.

    Parameters:
    lgbm (LGBMRegressor): Trained LightGBM model.
    X_test (np.array): Testing features.
    y_test (np.array): Testing target.

    Returns:
    tuple: Predicted values and R2 score.
    """
    y_pred = np.array(lgbm.predict(X_test))
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2


def _model_test_all(y_test, y_pred, r2):
    """
    Calculate overall evaluation metrics for the model.

    Parameters:
    y_test (list): List of testing targets.
    y_pred (list): List of predicted values.
    r2 (np.array): Array of R2 scores.

    Returns:
    tuple: Overall evaluation metrics.
    """
    y_test_all = np.concatenate(y_test, axis=None)
    y_pred_all = np.concatenate(y_pred, axis=None)
    r2_all = r2_score(y_test_all, y_pred_all)
    r2_mean = r2.mean()
    r2_std = r2.std()
    r2_sem = stats.sem(r2)
    r2_mean_clean = r2[r2 > 0].mean()
    r2_std_clean = r2[r2 > 0].std()
    r2_sem_clean = stats.sem(r2[r2 > 0])
    inexplicable = len(r2[r2 < 0])
    return r2_all, r2_mean, r2_std, r2_sem, r2_mean_clean, r2_std_clean, r2_sem_clean, inexplicable


class MultiModelling:
    """
    A class for multi-model training and evaluation.

    Attributes:
    df (pd.DataFrame): Input DataFrame.
    features_sets (list): List of feature sets.
    cortical_structures (np.array): Array of unique cortical structures.
    atlas (str): Atlas identifier.
    X_train (np.array): Array to store training features.
    X_test (np.array): Array to store testing features.
    y_train (np.array): Array to store training targets.
    y_test (np.array): Array to store testing targets.
    lgbm (np.array): Array to store trained models.
    y_pred (np.array): Array to store predicted values.
    r2 (np.array): Array to store R2 scores.
    models_evaluation (np.array): Array to store model evaluation metrics.
    df_eval_structures (pd.DataFrame): DataFrame to store evaluation metrics for structures.
    df_eval_features (pd.DataFrame): DataFrame to store evaluation metrics for feature sets.
    """

    def __init__(self, df, features_sets, atlas):
        """
        Initializes the MultiModelling class with the given parameters.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        features_sets (list): List of feature sets.
        atlas (str): Atlas identifier.
        """
        self.df = df
        self.features_sets = features_sets
        self.cortical_structures = np.unique(np.sort([int(x[1:]) for x in df.h_structure.unique()]))
        self.atlas = atlas    
        
        # data storage
        shape = (len(features_sets), len(self.cortical_structures))
        self.X_train = np.empty(shape, dtype=object)
        self.X_test = np.empty(shape, dtype=object) 
        self.y_train = np.empty(shape, dtype=object) 
        self.y_test = np.empty(shape, dtype=np.ndarray) 
        self.lgbm = np.empty(shape, dtype=object)
        self.y_pred = np.empty(shape, dtype=object) 
        self.r2 = np.empty(shape, dtype=object)
        self.models_evaluation = np.empty((len(features_sets), 8))
        
        # models
        for i, features in tqdm(enumerate(self.features_sets)):      
            for j, structure in enumerate(self.cortical_structures):
                self.X_train[i, j], self.X_test[i, j], self.y_train[i, j], self.y_test[i, j] = _split_train_test(self.df, features, structure)
                self.lgbm[i, j] = _model_train(self.X_train[i, j], self.y_train[i, j])
                self.y_pred[i, j], self.r2[i, j] = _model_test(self.lgbm[i, j], self.X_test[i, j], self.y_test[i, j])
            self.models_evaluation[i] = _model_test_all(self.y_test[i], self.y_pred[i], self.r2[i]) 
        
        # all evaluation
        self._evaluation_structures()
        self._evaluation_features()

    def _evaluation_structures(self):
        """
        Evaluate the model performance for each structure.

        Returns:
        None: Updates the df_eval_structures attribute.
        """
        df_eval_1 = pd.DataFrame([self.cortical_structures, self.r2[0], self.r2[1], self.r2[2]]).T
        df_eval_1.columns = 'atlas', 'r2_spatial', 'r2_temporal', 'r2_complete'
        df_eval_1 = pd.merge(df_eval_1, self.df[['structure_name', 'atlas']].drop_duplicates(), left_on="atlas", right_on="atlas")
        col_r2 = ['r2_spatial', 'r2_temporal', 'r2_complete']
        df_eval_1[col_r2] = df_eval_1[col_r2].where(df_eval_1[col_r2] > 0, -0.01)
        self.df_eval_structures = df_eval_1

    def _evaluation_features(self):
        """
        Evaluate the model performance for each feature set.

        Returns:
        None: Updates the df_eval_features attribute.
        """
        df_eval_2 = pd.DataFrame(self.models_evaluation)
        df_eval_2['features_set'] = self.features_sets
        df_eval_2.columns = 'r2_all', 'r2_mean', 'r2_std', 'r2_sem', 'r2_mean_clean', 'r2_std_clean', 'r2_sem_clean', 'inexplicable', 'features_set'
        df_eval_2['labels'] = ['Spatial Features', 'Temporal Features', 'Temporal and Spatial Features']
        self.df_eval_features = df_eval_2

    def evaluation_results(self, vertical=False):
        """
        Plot the evaluation results.

        Parameters:
        vertical (bool): Whether to plot the results vertically.

        Returns:
        None: Displays the evaluation plots.
        """
        df_eval_1 = self.df_eval_structures
        df_eval_2 = self.df_eval_features
        # plot 1A
        df_eval_1 = df_eval_1.sort_values(by='r2_complete', ascending=False).reset_index()
        plt.figure(figsize=(20, 5), dpi=80)
        plt.bar(df_eval_1.index - 0.2, df_eval_1['r2_spatial'], width=0.2, label='Spatial Features', color='purple', alpha=1)
        plt.bar(df_eval_1.index + 0.0, df_eval_1['r2_temporal'], width=0.2, label='Temporal Features', color='#a17fc0', alpha=1)
        plt.bar(df_eval_1.index + 0.2, df_eval_1['r2_complete'], width=0.2, label='Temporal and Spatial Features', color='#e75f00', alpha=1)
        plt.xticks(df_eval_1.index, df_eval_1.structure_name, rotation=45, horizontalalignment='right')
        plt.legend(loc='upper right')
        plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
        plt.show()
        # plot 1B
        if vertical:
            plt.figure(figsize=(5, 20), dpi=80)
            plt.barh(-df_eval_1.index + 0.2, df_eval_1['r2_spatial'], height=0.2, label='Spatial Features', color='purple', alpha=1)
            plt.barh(-df_eval_1.index + 0.0, df_eval_1['r2_temporal'], height=0.2, label='Temporal Features', color='#a17fc0', alpha=1)
            plt.barh(-df_eval_1.index - 0.2, df_eval_1['r2_complete'], height=0.2, label='Temporal and Spatial Features', color='#e75f00', alpha=1)
            plt.yticks(-df_eval_1.index, df_eval_1.structure_name, horizontalalignment='right')
            plt.legend()
            plt.grid(axis='x', color='0.7', linestyle='--', linewidth=1)
            plt.show()
        # plot 2
        plt.figure(figsize=(3, 5), dpi=80)
        bar = plt.bar(df_eval_2.labels, np.round(df_eval_2.r2_mean_clean, 2), color=['purple', '#a17fc0', '#e75f00'])
        plt.bar_label(bar, padding=10)
        plt.errorbar(df_eval_2.labels, df_eval_2.r2_mean_clean, yerr=df_eval_2.r2_sem_clean, fmt=".", color="black")
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.show()

        
    def summarize_info(self, path=[]):
        """
        Summarize the information and save it to a CSV file.

        Parameters:
        path (str): Path to save the CSV file.

        Returns:
        pd.DataFrame: Summarized DataFrame if path is not provided.
        """
        df_summarize = self.df[
            ['h_structure', 'atlas', 'Hemisphere', 'Lobe', 'Thickness at 10y.o.', 'Thickness at 80y.o.', 'Curvature',
             'Layer I thickness', 'Layer II thickness', 'Layer III thickness', 'Layer IV thickness', 'Layer V thickness', 
             'Layer VI thickness', 'bigbrain_layer_1', 'bigbrain_layer_2', 'bigbrain_layer_3', 'bigbrain_layer_4', 
             'bigbrain_layer_5', 'bigbrain_layer_6']].groupby(['h_structure']).mean()
        df_summarize['Global thinning'] = df_summarize['Thickness at 10y.o.'] - df_summarize['Thickness at 80y.o.']
        df_summarize['Global thinning normalized'] = (df_summarize['Thickness at 10y.o.'] - df_summarize['Thickness at 80y.o.']) / df_summarize['Thickness at 10y.o.']
        df_summarize = pd.merge(df_summarize, self.df_eval_structures, left_on="atlas", right_on="atlas")
        if path != []:
            df_summarize.to_csv(path + self.atlas + '_summary_data.csv')
        else:
            return df_summarize
