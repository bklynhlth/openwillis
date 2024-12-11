# author:    Kieran McVeigh
# website:   http://www.bklynhlth.com
from sklearn.mixture import GaussianMixture
import pandas as pd

def get_summary(df, start_at_col):
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the summary measurements from the framewise displacement data.

    Parameters:
    ............
    df : pandas.DataFrame
        framewise euclidean displacement dataframe
    start_at_col : int
        column index to start summary calculations

    Returns:
    ............
    df_summ : pandas.DataFrame
         stat summary dataframe

    ---------------------------------------------------------------------------------------------------
    """

    df_summ = pd.DataFrame()
    if len(df.columns)>0:
        df_mean = pd.DataFrame(df.mean()).T.iloc[:, start_at_col:].add_suffix('_mean')
        df_std = pd.DataFrame(df.std()).T.iloc[:, start_at_col:].add_suffix('_std')

        df_summ = pd.concat([df_mean, df_std], axis =1).reset_index(drop=True)
    return df_summ

def get_fps(df):
    """
    ---------------------------------------------------------------------------------------------------
    Calculate the frames per second (FPS) from a DataFrame.

    This function computes the FPS by taking the reciprocal of the mode of the time differences between consecutive rows in the DataFrame.

    Parameters:
    df : DataFrame
        A DataFrame containing a 'time' column with timestamps.

    Returns:
    int: The calculated frames per second (FPS).
    ---------------------------------------------------------------------------------------------------
    """
    #check if df.time exists
    if 'time' not in df.columns:
        raise ValueError('DataFrame must contain a time column.')
    return int(1/df.time.diff().mode())

def get_speaking_probabilities(df, rolling_std_seconds):
    """
    ---------------------------------------------------------------------------------------------------
    Calculate the probability of speaking at each frame in a DataFrame.

    This function calculates the probability of speaking at each frame in a DataFrame by fitting a Gaussian Mixture Model to the rolling standard deviation of the 'mouth_openness' column.

    Parameters:
    df : pd.DataFrame
        A DataFrame containing a 'time' column with timestamps and a 'mouth_openness' column with mouth openness values.
    rolling_std_seconds : int
        The number of seconds over which to calculate the rolling standard deviation.

    Returns:
    pandas.Series: A Series containing the probability of speaking at each frame.
    ---------------------------------------------------------------------------------------------------
    """
    fps = get_fps(df)
    rolling_std_frames = int(rolling_std_seconds*fps)
    df = df.copy(deep=True)
    df['rolling_mouth_open_std'] = df.mouth_openness.rolling(
        rolling_std_frames,
        min_periods=2
    ).std()


    df_nona = df[['frame','time','rolling_mouth_open_std']].dropna()
    gmm = GaussianMixture(n_components=2)
    gmm.fit(df_nona.rolling_mouth_open_std.values.reshape(-1,1))
    prob_preds = gmm.predict_proba(df_nona.rolling_mouth_open_std.values.reshape(-1,1))
    mean_1, mean_2 = gmm.means_.flatten()
    if mean_1 > mean_2:
        df_nona['speaking'] = prob_preds[:,0]
    else:
        df_nona['speaking'] = prob_preds[:,1]
        
    df = df.merge(df_nona, on='frame', how='left')
    return df.speaking

def split_speaking_df(df_disp, speaking_col, start_at_col):
    """
    ---------------------------------------------------------------------------------------------------

    This function splits the displacement dataframe into two dataframes based on speaking probability.

    Parameters:
    ............
    df_disp : pandas.DataFrame
        displacement dataframe
    speaking_col : str
        speaking probability column name
    start_at_col : int
        column index to start summary calculations


    Returns:
    ............
    df_summ : pandas.DataFrame
        stat summary dataframe
    ---------------------------------------------------------------------------------------------------
    """
    speaking_df = df_disp[df_disp[speaking_col] > 0.5]
    not_speaking_df = df_disp[df_disp[speaking_col] <= 0.5]
    speaking_df = speaking_df.drop(speaking_col, axis=1)
    not_speaking_df = not_speaking_df.drop(speaking_col, axis=1)

    speaking_df_summ = get_summary(speaking_df, start_at_col)
    not_speaking_df_summ = get_summary(not_speaking_df, start_at_col)
    speaking_df_summ = speaking_df_summ.add_suffix('_speaking')
    not_speaking_df_summ = not_speaking_df_summ.add_suffix('_not_speaking')
    
    df_summ = pd.concat([speaking_df_summ, not_speaking_df_summ], axis=1)
    
    return df_summ
