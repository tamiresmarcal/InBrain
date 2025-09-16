from boruta import BorutaPy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy import stats 


def corr_matriz(df):
    '''
    Calculate features correlation.
    parameter:
        df(dataframe): dataframe with only numeric continuous features.
        
    return: 
        plot: Correlation Matriz.
    '''
    f = plt.figure(figsize=(19, 15))
    s = 14
    data = df.corr()
    plt.matshow(data, fignum=f.number)
    for (x, y), value in np.ndenumerate(data):
        plt.text(x, y, f"{value:.2f}", va="center", ha="center")
    plt.xticks(range(df.shape[1]), df.columns, 
               fontsize=s, rotation=45, horizontalalignment = 'left')
    plt.yticks(range(df.shape[1]), df.columns, fontsize=s)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=s)
    plt.title('Correlation Matriz', fontsize=16);
    plt.show()

    
def boruta(df, target = 'annual_rate'):
    '''
    Execute Boruta test on a dataset.
    parameter:
        df(dataframe): dataframe with only numeric continuous features.
        target(string): target you want to predict.
        
    return: 
        print: Feature, Rank and Support.
    '''
    X = df.drop(columns =[target]) 
    y = df[target]
    forest = RandomForestRegressor(n_jobs=-1, max_depth=20)
    forest.fit(X, y)
    feat_selector = BorutaPy(forest, n_estimators='auto', random_state=1, alpha= 0.001)
    feat_selector.fit(X.to_numpy(), y)
    feature_ranks = list(zip(X.columns, 
                             feat_selector.ranking_, 
                             feat_selector.support_,
                            ))
    for feat in feature_ranks:
        print('Feature: {:<25} Rank: {},  Support: {}'.format(feat[0], feat[1], feat[2]))

        
def life_stages(a, temporal_threshold):
    if a<=temporal_threshold[0]:
        return 'Development'
    if a>=temporal_threshold[1]:
        return 'Aging'
    else:
        return 'MidLife'

    
def _structures_group(a, spatial_threshold):
    if a<= -spatial_threshold:
        return 'DifNeg'
    if a>= spatial_threshold:
        return 'DifPosi'
    else:
        return 'DifNeu'
    
    
def _build_life_stages_df(df, temporal_threshold = [30,60], spatial_threshold = 0.35):
    df_int = df[['annual_rate','Age', 'Curvature',
         'thickness', 'Thickness at 10y.o.', 'Thickness at 80y.o.', 'thickness_mean',  
         'Layer I thickness','Layer II thickness', 'Layer III thickness', 
         'Layer IV thickness','Layer V thickness', 'Layer VI thickness',
         'bigbrain_layer_1', 'bigbrain_layer_2', 'bigbrain_layer_3',
         'bigbrain_layer_4', 'bigbrain_layer_5', 'bigbrain_layer_6', 'atlas', 'structure_name']
       ]
    # averages thinning in 3 life stages
    avg_rate0 = df_int[df_int.Age<=temporal_threshold[0]].groupby(by='structure_name').annual_rate.mean()
    avg_rate0 = avg_rate0/avg_rate0.mean()
    avg_rate1 = df_int[(df_int.Age<temporal_threshold[1]) & (df_int.Age>temporal_threshold[0])].groupby(by='structure_name').annual_rate.mean()
    avg_rate1 = avg_rate1/avg_rate1.mean()
    avg_rate2 = df_int[df_int.Age>=temporal_threshold[1]].groupby(by='structure_name').annual_rate.mean()
    avg_rate2 = avg_rate2/avg_rate2.mean()
    # dataframe averages
    df_life_stages = pd.concat([avg_rate0,avg_rate1,avg_rate2], axis=1)
    df_life_stages.columns = ['Development Annual Rate','Mid Life Annual Rate','Aging Annual Rate']
    df_life_stages = pd.concat([df_int.groupby(by='structure_name').mean(),df_life_stages], axis=1)
    df_life_stages['Development Aging Difference'] = df_life_stages['Development Annual Rate'] - df_life_stages['Aging Annual Rate']
    df_life_stages['StructuresGroup'] = df_life_stages['Development Aging Difference'].apply(lambda x : _structures_group(x, spatial_threshold = spatial_threshold))
    df_life_stages = df_life_stages.drop(columns=['Age'])
    return df_life_stages


def violin_life_stages(df):
    sns.violinplot(data=df, x="annual_rate", y="LifeStage", palette='magma')
    plt.grid(alpha = 0.2, linestyle='--', which='both')
    plt.show()
    

def corr_life_stages(df, temporal_threshold = [30,60], spatial_threshold = 0.35):
    df_life_stages = _build_life_stages_df(df, temporal_threshold, spatial_threshold)                                                                                       
    f = plt.figure(figsize=(19, 15))
    s = 12
    col = ['Development Aging Difference','Development Annual Rate','Mid Life Annual Rate','Aging Annual Rate']
    test = ['annual_rate', 'Development Annual Rate', 'Mid Life Annual Rate', 'Aging Annual Rate',
       'Development Aging Difference',
       'Curvature', 'thickness', 'Thickness at 10y.o.',
       'Thickness at 80y.o.', 'thickness_mean', 'Layer I thickness',
       'Layer II thickness', 'Layer III thickness', 'Layer IV thickness',
       'Layer V thickness', 'Layer VI thickness', 'bigbrain_layer_1',
       'bigbrain_layer_2', 'bigbrain_layer_3', 'bigbrain_layer_4',
       'bigbrain_layer_5', 'bigbrain_layer_6']
    data = df_life_stages.corr().loc[test,col]
    plt.matshow(data.T, fignum=f.number)
    for (x, y), value in np.ndenumerate(data):
        plt.text(x, y, f"{value:.2f}", va="center", ha="center")
    plt.yticks(range(len(col)), col,fontsize=s)
    plt.xticks(range(len(data)), test, fontsize=s, rotation=45, horizontalalignment = 'left')
    plt.title('Correlation Life Stages', fontsize=16);
    plt.show()
    
    
def differenceplot(df, temporal_threshold = [30,60], spatial_threshold = 0.35, x="bigbrain_layer_1"):
    df_life_stages = _build_life_stages_df(df, temporal_threshold, spatial_threshold)
    data = df_life_stages.sort_values(by = 'Development Aging Difference')
    plt.figure(figsize=(16, 4), dpi=80)
    plt.plot(data.index.astype(str), data['Aging Annual Rate'], '--o', color='#ff9135') 
    plt.plot(data.index.astype(str), data['Development Annual Rate'], '--o', color='#280069')
    plt.grid(alpha = 0.2, linestyle='--', which='both')
    plt.title('Development and Aging Annual Thinning Rates sorted by its Difference')
    plt.xticks(rotation = 45, horizontalalignment = 'right') 
    plt.ylabel("Average thinning rates")
    plt.show()
    

def joinplot(df, temporal_threshold = [30,60], spatial_threshold = 0.35, x="bigbrain_layer_1"):
    df_life_stages = _build_life_stages_df(df, temporal_threshold, spatial_threshold)
    y='Development Aging Difference'
    corr, p_value = stats.pearsonr(df_life_stages[x], df_life_stages[y])
    r_square = round(corr**2,3)
    g = sns.jointplot(data=df_life_stages, x=x, y=y, kind='reg', color='#280069')
    plt.annotate(f'r = {corr:.3f}', xy=(0.8, 0.95), xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
    plt.xlabel('Layer I Percentage')
    plt.show()
    
    
def find_intervals(df, x="bigbrain_layer_1", temporal_threshold = [30,60], spatial_threshold = 0.35):
    df_life_stages = _build_life_stages_df(df, temporal_threshold, spatial_threshold)
    #test
    g1 = df_life_stages[df_life_stages['StructuresGroup'] =='DifNeg']
    g2 = df_life_stages[df_life_stages['StructuresGroup'] =='DifNeu']
    g3 = df_life_stages[df_life_stages['StructuresGroup'] =='DifPosi']
    pkruskal = stats.kruskal(g1[x],g2[x],g3[x])
    pf_oneway = stats.f_oneway(g1[x],g2[x],g3[x])
    #df
    data = [df_life_stages[['bigbrain_layer_1','StructuresGroup']].groupby('StructuresGroup').apply(lambda x: len(x)),
            df_life_stages[['bigbrain_layer_1','StructuresGroup']].groupby('StructuresGroup').apply(lambda x: np.mean(x)).round(3),
            pd.DataFrame(df_life_stages[['bigbrain_layer_1','StructuresGroup']].groupby('StructuresGroup').apply(lambda x: stats.sem(x)[0])).round(3),
            df_life_stages[['bigbrain_layer_1','StructuresGroup']].groupby('StructuresGroup').apply(lambda x: np.std(x)).round(3)]
    df_groups = pd.concat(data, axis=1)
    df_groups.columns=['n_structures',x+'_mean',x+'_sem',x+'_std']
    return df_groups, pkruskal, pf_oneway


def violinplot(df, temporal_threshold = [30,60], spatial_threshold = 0.35, x="bigbrain_layer_1"):
    df_life_stages = _build_life_stages_df(df, temporal_threshold, spatial_threshold)
    # Discretization
    sns.violinplot(data=df_life_stages, x=x, y="StructuresGroup", palette='magma', scale='count', order=['DifPosi','DifNeu','DifNeg'])
    #sns.swarmplot(data=df_life_stages, x=x, y="StructuresGroup", alpha = 0.5, order=['DifPosi','DifNeu','DifNeg'])
    plt.xlabel('Layer I Percentage')
    plt.show()
