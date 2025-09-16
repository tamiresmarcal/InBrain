import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns


class ShapExplicability:
    """ Explicability for multimodel aproach with Shap Values.
        Parameters:
            X_train(np.array): X_train set
            lgmb(object): models set
            df_eval_1(dataframe): explicables features indentifier
            features_sets(list): features sets
            features_set(int): model indentifier
        Return:
            summary_plot viollin and summary_plot bar
    """     

    def __init__(self, X_train, lgbm, df_eval_1, features_sets, features_set=2):
        explicables_index = df_eval_1[df_eval_1.r2_complete>0].index
        i = explicables_index[0]
        X_train_ = pd.DataFrame(X_train[features_set,i])
        X_train_.columns = features_sets[features_set]
        shap_X = X_train_
        explainer = shap.Explainer(lgbm[features_set,i])
        shap_complete = explainer(X_train_)
        shap_interaction = explainer.shap_interaction_values(X_train_)
        for i in tqdm(explicables_index[1:]):
            # data
            X_train_ = pd.DataFrame(X_train[features_set,i])
            X_train_.columns = features_sets[features_set]
            # shap computation
            explainer = shap.Explainer(lgbm[features_set,i])
            shap_values = explainer(X_train_)
            shap_values_interaction = explainer.shap_interaction_values(X_train_)
            # save
            shap_complete.values = np.append(shap_complete.values, shap_values.values,axis=0)
            shap_complete.base_values = np.append(shap_complete.base_values, shap_values.base_values, axis=0)
            shap_complete.data = np.append(shap_complete.data, shap_values.data, axis=0)
            shap_interaction = np.append(shap_interaction, shap_values_interaction ,axis=0)
            shap_X = np.append(shap_X, X_train_, axis=0)
     
        self.shap_complete = shap_complete
        self.shap_interaction = shap_interaction
        self.features = features_sets[features_set]
        self.shap_X = shap_X

    def summary(self):
        """ Plot visualization to summary shap information.
            Parameters:
                self: shap_complete
            Return:
                summary_plot viollin and summary_plot bar
        """    
        shap.summary_plot(self.shap_complete)
        shap.summary_plot(self.shap_complete, plot_type='bar',color='purple')

        
    def scatter(self, feature, cmap=plt.cm.magma):
        """ Plot all features contribution colorcoded by any feature.
            Parameters:
                feature(str): model feature
                cmap(str): color pallete
                self: shap_complete
            Return:
                all features contribution plot with interaction colorcoded by args feature.
        """
        for i in range(len(self.features)):
            shap.plots.scatter(self.shap_complete[:, i], 
                               color=self.shap_complete[:,feature], dot_size=5, cmap=cmap, alpha =0.8)

    def interaction_matrices(self):
        """ Plot features interaction based on average.
            Parameters:
                self: shap_interaction, shap_X, features
            Return:
                features interaction matrices in heatmap format and shap summary format
        """
        # Get absolute mean of matrices
        mean_shap = np.abs(self.shap_interaction).mean(0)*10000
        df_mean = pd.DataFrame(mean_shap,index=self.features,columns=self.features)
        # times off diagonal by 2
        df_mean.where(df_mean.values == np.diagonal(df_mean),df_mean.values*2,inplace=True)
        # display heatmap
        plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
        plt.title('Shap interaction mean')
        sns.set(font_scale=0.8)
        sns.heatmap(df_mean,cmap='coolwarm',annot=True,fmt='.3g',cbar=False)
        plt.xticks(rotation=45)
        plt.show()
        # display shap summary plot
        shap.summary_plot(self.shap_interaction, 
                          features=np.array(self.shap_X), 
                          feature_names =np.array(self.features),
                          max_display=12) 
        
 

