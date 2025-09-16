from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import preprocessing as pp
import warnings
warnings.filterwarnings('ignore')

def read_database(atlas, paths):
    """
    Reads and processes the MRI data based on the specified atlas.

    Parameters:
    atlas (str): The atlas to use ('Economo' or 'DKT').
    paths (dict): Dictionary containing paths to the data files.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    mri_subs_path_v2 = paths['data_dkt']
    dict_dkt_path = paths['dict_dkt']
    mri_subs_path_v1 = paths['data_economo']
    dict_econo_path = paths['dict_economo']
    bb_path = paths['data_bb']
    
    if atlas == 'Economo':
        # read df
        df = pd.read_parquet(mri_subs_path_v1) 
        df[['hemisphere', 'atlasEcono', 'handedness', 'sex', 'age']] = df[['hemisphere', 'atlasEcono', 'handedness', 'sex', 'age']].astype('int32')
        # add bb
        df = pp.bb_features(bb_path, df)
        df = df.dropna()
        # add name structures and lobes
        Dict_structures_econo = pd.read_csv(dict_econo_path, sep=';', header=None)
        df = df.merge(Dict_structures_econo, how='left', left_on='atlasEcono', right_on=0)
        df = df.rename(columns={1: 'lobe', 2: 'structure_name'})
        df = df.drop(columns=0)
        df = df.rename(columns={'atlasEcono': 'atlas'})
    
    if atlas == 'DKT':
        # read df
        df = pd.read_parquet(mri_subs_path_v2) 
        df = df[df.atlasDF != -1.0]
        df = df.rename(columns={'h_atlasDF': 'h_structure'})
        # add name structures and lobes
        Dict_structures_dkt = pd.read_csv(dict_dkt_path, sep=',', header=None)
        Dict_structures_dkt = Dict_structures_dkt.rename(columns={2: 'structure_name', 3: 'lobe_name', 4: 'lobe'})
        Dict_structures_dkt = Dict_structures_dkt.drop(columns=[0, 1])
        df = df.join(Dict_structures_dkt, on='atlasDF')
        df = df.rename(columns={'atlasDF': 'atlas'})
        df = df[['participant', 'age', 'sex', 'handedness', 'hemisphere', 'lobe_name', 'lobe', 'structure_name', 'atlas', 'h_structure',  
                 'bigbrain_layer_1', 'bigbrain_layer_2', 'bigbrain_layer_3', 'bigbrain_layer_4', 'bigbrain_layer_5', 'bigbrain_layer_6', 
                 'area', 'curv', 'sulc', 'thickness']]
    return df

    
def reg_polynomial(df, plots=False):
    """
    Fits a polynomial regression model to the data and calculates the annual rate of change in thickness.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    plots (bool): Whether to plot the regression results.

    Returns:
    pd.DataFrame: DataFrame containing the annual rate of change in thickness for each structure.
    """
    h_structures = np.sort(df.h_structure.unique())
    taxa_anual = []
    for h_structure in h_structures:
        age = df[df.h_structure == h_structure].age
        thickness = df[df.h_structure == h_structure].thickness
        z = np.polyfit(age, thickness, 3)
        p = np.poly1d(z)
        xp = np.linspace(10, 80, 100)
        p2 = np.polyder(p)
        derivada = [-p2(year) for year in range(0, 100)]
        taxa_anual.append([h_structure] + derivada)
        if plots:
            plots_reg_polynomial(h_structure, df, age, thickness, xp, p, derivada)
    if not plots:
        taxa_anual = pd.DataFrame(taxa_anual)
        taxa_anual = taxa_anual.rename(columns={0: "atlas"})
        taxa_anual.set_index('atlas', inplace=True)
        return taxa_anual


def _rate(taxa_anual, h_structure, age):
    """
    Helper function to get the annual rate of change in thickness for a given structure and age.

    Parameters:
    taxa_anual (pd.DataFrame): DataFrame containing the annual rate of change in thickness.
    h_structure (int): Structure identifier.
    age (int): Age.

    Returns:
    float: Annual rate of change in thickness.
    """
    return taxa_anual.loc[h_structure, age]


def _reference_thickness_values(df, year=10):
    """
    Helper function to calculate the reference thickness values for a given year.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    year (int): Year for which to calculate the reference thickness values.

    Returns:
    pd.DataFrame: DataFrame with the reference thickness values added.
    """
    thickness = df[df.age == year][['h_structure', 'thickness']].groupby(['h_structure']).mean()
    thickness = thickness.rename(columns={'thickness': 'Thickness at ' + str(year) + 'y.o.'})
    thickness = thickness.round(2)
    df = df.merge(thickness, on='h_structure')
    return df

    
def _mean_thickness_values(df):
    """
    Helper function to calculate the mean thickness values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with the mean thickness values added.
    """
    thickness_mean = df[['h_structure', 'thickness']].groupby(['h_structure']).mean()
    thickness_mean = thickness_mean.rename(columns={'thickness': 'thickness_mean'})
    thickness_mean = thickness_mean.round(2)
    df = df.merge(thickness_mean, on='h_structure')
    return df


def _layers_thicknesses(df):
    """
    Helper function to calculate the thickness values for each layer.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with the thickness values for each layer added.
    """
    df['thickness_layer_1'] = df.thickness * df.bigbrain_layer_1
    df['thickness_layer_2'] = df.thickness * df.bigbrain_layer_2
    df['thickness_layer_3'] = df.thickness * df.bigbrain_layer_3
    df['thickness_layer_4'] = df.thickness * df.bigbrain_layer_4
    df['thickness_layer_5'] = df.thickness * df.bigbrain_layer_5
    df['thickness_layer_6'] = df.thickness * df.bigbrain_layer_6
    df['sum_thickness_layers'] = df['thickness_layer_1'] + df['thickness_layer_2'] + df['thickness_layer_3'] + df['thickness_layer_4'] + df['thickness_layer_5'] + df['thickness_layer_6']
    return df


def _groupby_age_structure(df):
    """
    Helper function to group the data by age and structure.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Grouped DataFrame.
    """
    dfs = []
    for h_structure in df.h_structure.unique():
        df0 = df[df.h_structure == h_structure].groupby(['age']).mean()
        df0['h_structure'] = h_structure
        df0['structure_name'] = np.array(df[df.h_structure == h_structure].structure_name[:1])[0]
        dfs.append(df0.reset_index())
    return pd.concat(dfs).reset_index(drop=True)


def build_main_dataframe(df, taxa_anual):
    """
    Builds the main DataFrame with the calculated annual rates and other features.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    taxa_anual (pd.DataFrame): DataFrame containing the annual rate of change in thickness.

    Returns:
    pd.DataFrame: Main DataFrame with the calculated features.
    """
    df1 = df.copy()
    df1['anual_rate'] = df1.apply(lambda x: _rate(taxa_anual, x.h_structure, x.age), axis=1)
    df1 = _reference_thickness_values(df1, year=10)
    df1 = _reference_thickness_values(df1, year=80)
    df1 = _mean_thickness_values(df1)
    df1 = _layers_thicknesses(df1)
    df1 = _groupby_age_structure(df1)
    df1.rename(columns={'hemisphere': 'Hemisphere', 'sex': 'Gender', 'age': 'Age', 'lobe': 'Lobe', 'thickness_layer_1': 'Layer I thickness',
                        'thickness_layer_2': 'Layer II thickness', 'thickness_layer_3': 'Layer III thickness', 'thickness_layer_4': 'Layer IV thickness',
                        'thickness_layer_5': 'Layer V thickness', 'thickness_layer_6': 'Layer VI thickness', 'sulc': 'Curvature'}, inplace=True)
    return df1


def plots_reg_polynomial(h_structure, df, age, thickness, xp, p, derivada):
    """
    Plots the polynomial regression results.

    Parameters:
    h_structure (int): Structure identifier.
    df (pd.DataFrame): Input DataFrame.
    age (pd.Series): Age series.
    thickness (pd.Series): Thickness series.
    xp (np.array): X values for plotting the polynomial.
    p (np.poly1d): Polynomial function.
    derivada (list): List of derivative values.

    Returns:
    None: Displays the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    a = df[df.h_structure == h_structure].structure_name.iloc[0] 
    b = ' | ' + h_structure[1:]
    c = ' | hemisphere ' + str(int(df[df.h_structure == h_structure].hemisphere.iloc[0]))
    fig.suptitle(a + b + c)
    ax1.plot(age, thickness, '.', markersize=15, color='0.6', alpha=0.2)
    ax1.plot(xp, p(xp), '-', linewidth=2, color='purple')
    ax1.axis(ymin=1, ymax=4.5)
    ax2.plot(derivada, linewidth=2, color='purple')
    ax2.axis(ymin=-0.005, ymax=0.035)
    ax2.axis(xmin=10, xmax=85)
    

def plot_avarage_thinning_rates(df):
    """
    Plots the average thinning rates.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    None: Displays the plots.
    """
    age = df.age
    thickness = df.thickness
    z = np.polyfit(age, thickness, 3)
    p = np.poly1d(z)
    xp = np.linspace(10, 80, 100)
    p2 = np.polyder(p)
    derivada = [-p2(year) for year in range(0, 100)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    ax1.plot(age, thickness, '.', markersize=8, color='0.6', alpha=0.2)
    ax1.plot(xp, p(xp), '-', linewidth=2, color='purple')    
    ax1.axis(ymin=1, ymax=4.5)
    ax2.plot(derivada, color='purple')
    ax2.axis(ymin=-0.005, ymax=0.035)
    ax2.axis(xmin=10, xmax=85)
    plt.figure(figsize=(10, 1), dpi=80)
    plt.hist(df.age, color='0.9')
    plt.show()


def find_positive_structures(df1, rate=-0.003):
    """
    Finds structures with positive thickness variation.

    Parameters:
    df1 (pd.DataFrame): Input DataFrame.
    rate (float): Threshold rate for positive variation.

    Returns:
    pd.DataFrame: DataFrame containing structures with positive thickness variation.
    """
    positives = df1[['structure_name', 'atlas', 'Hemisphere', 'Age', 'anual_rate']]
    print("Percentage of thickness positive variation: " + str(len(positives[positives.anual_rate < rate]) / len(positives)))
    print("Structures of thickness positive variation: " + str(positives[positives.anual_rate < rate].structure_name.unique()))
    return positives[positives.anual_rate < rate]


def plot_all_thinning(df1):
    """
    Plots the thinning rates for all structures.

    Parameters:
    df1 (pd.DataFrame): Input DataFrame.

    Returns:
    None: Displays the plots.
    """
    plt.scatter(df1.Age, df1.anual_rate, alpha=0.4, c=df1.Age, cmap=plt.cm.magma, s=10)
    plt.show()
    plt.scatter(df1.Age, df1.anual_rate, alpha=0.4, c=df1.Lobe, cmap=plt.cm.viridis, s=10)
    plt.show()
