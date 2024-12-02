import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from itertools import combinations
from scipy.fft import fft
from scipy.stats import entropy


def get_dominant_position(df):
    position_counts = df['position'].value_counts()
    most_common_position = position_counts.index[0]
    if pd.isna(most_common_position):
        most_common_position = position_counts.index[1]
        print("NaN dominant")
    return most_common_position
        
        
def correlate_robust_mod(data, prefix):
    try:
        if data.shape[1] < 2:
            raise ValueError("DataFrame must contain at least two columns for correlation calculation.")
        corr_matrix = data.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlations = upper.stack().reset_index()
        correlations.columns = ['column1', 'column2', 'correlation']
        correlations['correlation'] = correlations['correlation'].astype(float)
        if correlations.empty:
            raise ValueError("No valid correlations could be calculated.")
        num_columns = len(data.columns)
        num_combinations = int(num_columns * (num_columns - 1) / 2)
        flattened = np.empty(num_combinations)
        flattened[:] = np.nan
        
        for i, (col1, col2) in enumerate(combinations(data.columns, 2)):
            filtered_idx = correlations[(correlations['column1'] == col1) & 
                                        (correlations['column2'] == col2)].index
            if filtered_idx.empty:
                raise ValueError(f"No correlation found for the pair: {col1}, {col2}")
            idx = i if len(data.columns) == 2 else filtered_idx[0]
            flattened[idx] = correlations.loc[idx, 'correlation']
        return pd.DataFrame([flattened], columns=[f'{prefix}_{x[0]}_{x[1]}' for x in combinations(data.columns, 2)])
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return pd.DataFrame()


def motion_features_mod(ds_t, start, end, complete=True, positions=[]):
    # Non-feature info about a window
    code_out = ds_t[['time_point', 'baby', 'position', 'study']].iloc[0].to_frame().T
    code_out['start'] = start
    code_out['end'] = end
    position_counts = ds_t['position'].value_counts()
    dominant_position = position_counts.index[0]
    code_out['position'] = dominant_position
    if dominant_position in positions:
        # Create a subset of data with only the sensor data
        ds_t = ds_t.drop(columns=['time_point', 'baby', 'position', 'times [s]', 'study'])
        ds_t = ds_t.drop(columns=[col for col in ds_t.columns if 'arm' in col and 'AC' in col])
        features = {}
        # Creating frequency features
        for col in ds_t.columns:
            x = ds_t[col].values
            x_prime = x - np.mean(x)
            N = len(x)
            f_transform = fft(x_prime)
            x_hat = np.abs(f_transform[:N//2+1])
            E = np.sum(x_hat ** 2)
            p = x_hat / np.sum(x_hat)
            H = -np.sum(p * np.log(p + 1e-12))/(np.log2(N/2 + 1e-12))
            xi = np.arange(N//2+1)
            c = np.sum(xi * p)
            delta = xi - c
            b = np.sum(delta * p)
            max_freq_index = np.argmax(x_hat)
            features[col + '_energy'] = E
            features[col + '_entropy'] = H
            features[col + '_centroid'] = c
            features[col + '_bandwidth'] = b
            features[col + '_max_freq'] = max_freq_index
        freq_features = pd.DataFrame([features])
        freq_features.index = code_out.index

        # Defining summary statistics functions
        summary_functions = {'MEAN': lambda x: np.mean(x), 'SD': lambda x: np.std(x), 'MED': lambda x: np.median(x), 'SKEW': lambda x: skew(x),
                             'KURT': lambda x: kurtosis(x), 'MIN': lambda x: np.min(x), 'MAX': lambda x: np.max(x),
                             'P25': lambda x: np.quantile(x, q=0.25), 'P75': lambda x: np.quantile(x, q=0.75)}
        # Calculating summary stats for each sensor
        mot_features = pd.DataFrame({stat + '_' + col: func(ds_t[col]) for stat, func in summary_functions.items() for col in ds_t.columns}, index=[0])
        mot_features.index = code_out.index

        # Creating other time-domain features
        # Commented out lines were used for 6 sensor locations
        if complete:
            # Calculate differences between sensor readings
            diff_fx_axes = {'DIFF12': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 1]), index=[0]),
                        'DIFF13': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 2]), index=[0]),
                        'DIFF14': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 3]), index=[0]),
                        'DIFF15': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 4]), index=[0]),
                        'DIFF16': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 5]), index=[0]),
                        'DIFF23': lambda x: pd.Series(np.mean(x.iloc[:, 1] - x.iloc[:, 2]), index=[0]),
                        'DIFF24': lambda x: pd.Series(np.mean(x.iloc[:, 1] - x.iloc[:, 3]), index=[0]),
                        'DIFF25': lambda x: pd.Series(np.mean(x.iloc[:, 1] - x.iloc[:, 4]), index=[0]),
                        'DIFF26': lambda x: pd.Series(np.mean(x.iloc[:, 1] - x.iloc[:, 5]), index=[0]),
                        'DIFF34': lambda x: pd.Series(np.mean(x.iloc[:, 2] - x.iloc[:, 3]), index=[0]),
                        'DIFF35': lambda x: pd.Series(np.mean(x.iloc[:, 2] - x.iloc[:, 4]), index=[0]),
                        'DIFF36': lambda x: pd.Series(np.mean(x.iloc[:, 2] - x.iloc[:, 5]), index=[0]),
                        'DIFF45': lambda x: pd.Series(np.mean(x.iloc[:, 3] - x.iloc[:, 4]), index=[0]),
                        'DIFF46': lambda x: pd.Series(np.mean(x.iloc[:, 3] - x.iloc[:, 5]), index=[0]),
                        'DIFF56': lambda x: pd.Series(np.mean(x.iloc[:, 4] - x.iloc[:, 5]), index=[0])}
            diff_fx_sensors = {'DIFF12': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 1]), index=[0]),
                            'DIFF13': lambda x: pd.Series(np.mean(x.iloc[:, 0] - x.iloc[:, 2]), index=[0]),
                            'DIFF23': lambda x: pd.Series(np.mean(x.iloc[:, 1] - x.iloc[:, 2]), index=[0])}

            # Calculate summaries across axes for each sensor
            sensor_features = {
                #'AC_Acc_infantleftarm': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantleftarm' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]],
                #'AC_Acc_infantrightarm': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantrightarm' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]],
                'AC_Acc_infanttrunkmid': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infanttrunkmid' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]],
                'AC_Acc_infanttrunkleft': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infanttrunkleft' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]],
                'AC_Acc_infantleftleg': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantleftleg' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]],
                'AC_Acc_infantrightleg': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantrightleg' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]],
                'DC_Acc_infantleftarm': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantleftarm' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]],
                'DC_Acc_infantrightarm': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantrightarm' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]],
                'DC_Acc_infanttrunkmid': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infanttrunkmid' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]],
                'DC_Acc_infanttrunkleft': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infanttrunkleft' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]],
                'DC_Acc_infantleftleg': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantleftleg' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]],
                'DC_Acc_infantrightleg': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantrightleg' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]],
                #'Acc_infantleftarm': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantleftarm' in col and not any(s in col for s in ['module', 'variance', 'AC', 'DC'])]],
                #'Acc_infantrightarm': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantrightarm' in col and not any(s in col for s in ['module', 'variance', 'AC', 'DC'])]],
                'Acc_infanttrunkmid': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infanttrunkmid' in col and not any(s in col for s in ['module', 'variance', 'AC', 'DC'])]],
                'Acc_infanttrunkleft': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infanttrunkleft' in col and not any(s in col for s in ['module', 'variance', 'AC', 'DC'])]],
                'Acc_infantleftleg': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantleftleg' in col and not any(s in col for s in ['module', 'variance', 'AC', 'DC'])]],
                'Acc_infantrightleg': ds_t[[col for col in ds_t.columns if 'Acc' in col and 'infantrightleg' in col and not any(s in col for s in ['module', 'variance', 'AC', 'DC'])]],
                #'Gyr_infantleftarm': ds_t[[col for col in ds_t.columns if 'Gyr' in col and 'infantleftarm' in col]],
                #'Gyr_infantrightarm': ds_t[[col for col in ds_t.columns if 'Gyr' in col and 'infantrightarm' in col]],
                'Gyr_infanttrunkmid': ds_t[[col for col in ds_t.columns if 'Gyr' in col and 'infanttrunkmid' in col]],
                'Gyr_infanttrunkleft': ds_t[[col for col in ds_t.columns if 'Gyr' in col and 'infanttrunkleft' in col]],
                'Gyr_infantleftleg': ds_t[[col for col in ds_t.columns if 'Gyr' in col and 'infantleftleg' in col]],
                'Gyr_infantrightleg': ds_t[[col for col in ds_t.columns if 'Gyr' in col and 'infantrightleg' in col]],
                #'Mag_infantleftarm': ds_t[[col for col in ds_t.columns if 'Mag' in col and 'infantleftarm' in col]],
                #'Mag_infantrightarm': ds_t[[col for col in ds_t.columns if 'Mag' in col and 'infantrightarm' in col]],
                'Mag_infanttrunkmid': ds_t[[col for col in ds_t.columns if 'Mag' in col and 'infanttrunkmid' in col]],
                'Mag_infanttrunkleft': ds_t[[col for col in ds_t.columns if 'Mag' in col and 'infanttrunkleft' in col]]}#,
                #'Mag_infantleftleg': ds_t[[col for col in ds_t.columns if 'Mag' in col and 'infantleftleg' in col]],
                #'Mag_infantrightleg': ds_t[[col for col in ds_t.columns if 'Mag' in col and 'infantrightleg' in col]]}

            sensor_sums = pd.DataFrame({key + '_SUM': float(value.sum(axis=0).mean()) for key, value in sensor_features.items()}, index=[0])
            sensor_abssums = pd.DataFrame({key + '_MAG': float(value.abs().sum(axis=0).mean()) for key, value in sensor_features.items()}, index=[0])

            # Calculate correlations
            sensor_corrs = pd.concat([correlate_robust_mod(data, f'CORR_{name}') for name, data in sensor_features.items()], axis=1)
            sensor_abscorrs = pd.concat([correlate_robust_mod(data.abs(), f'ABSCORR_{name}') for name, data in sensor_features.items()], axis=1)

            # Calculate differences between sensor readings
            sensor_diffs = pd.concat({f"DIFF_{key}_{diff_key}": diff_func(value) for key, value in sensor_features.items() for diff_key, diff_func in diff_fx_sensors.items()}, axis=1)
            sensor_absdiffs = pd.concat({f"ABSDIFF_{key}_{diff_key}": diff_func(value.abs()) for key, value in sensor_features.items() for diff_key, diff_func in diff_fx_sensors.items()}, axis=1)

            # Calculate summaries across sensors for each axis
            cross_features = {'AC_XAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_X' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]),
                              'AC_YAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_Y' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]),
                              'AC_ZAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_Z' in col and 'AC' in col and not any(s in col for s in ['module', 'variance'])]),
                              'DC_XAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_X' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]),
                              'DC_YAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_Y' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]),
                              'DC_ZAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_Z' in col and 'DC' in col and not any(s in col for s in ['module', 'variance'])]),
                              'AC_Acc_module': ds_t.filter(items=[col for col in df.columns if 'AC_Acc_module' in col]),
                              'DC_Acc_module': ds_t.filter(items=[col for col in df.columns if 'DC_Acc_module' in col]),
                              'Acc_module': ds_t.filter(items=[col for col in df.columns if 'Acc_module' in col and not any(s in col for s in ['AC', 'DC'])]),
                              'roll': ds_t.filter(items=[col for col in df.columns if 'roll' in col]),
                              'pitch': ds_t.filter(items=[col for col in df.columns if 'pitch' in col]),
                              'XAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_X' in col and not any(fs in col for fs in ['module', 'variance', 'AC', 'DC'])]),
                              'YAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_Y' in col and not any(fs in col for fs in ['module', 'variance', 'AC', 'DC'])]),
                              'ZAcc': ds_t.filter(items=[col for col in df.columns if 'Acc_Z' in col and not any(fs in col for fs in ['module', 'variance', 'AC', 'DC'])]),
                              'XGyr': ds_t.filter(like='Gyr_X'),
                              'YGyr': ds_t.filter(like='Gyr_Y'),
                              'ZGyr': ds_t.filter(like='Gyr_Z'),
                              'XMag': ds_t.filter(like='Mag_X'),
                              'YMag': ds_t.filter(like='Mag_Y'),
                              'ZMag': ds_t.filter(like='Mag_Z')}

            sums = pd.DataFrame({key + '_SUM': float(value.sum(axis=0).mean()) for key, value in cross_features.items()}, index=[0])
            abssums = pd.DataFrame({key + '_MAG': float(value.abs().sum(axis=0).mean()) for key, value in cross_features.items()}, index=[0])

            corrs = pd.concat([correlate_robust_mod(data, f'CORR_{name}') for name, data in cross_features.items()], axis=1)
            abscorrs = pd.concat([correlate_robust_mod(data.abs(), f'ABSCORR_{name}') for name, data in cross_features.items()], axis=1)

            diffs = pd.concat({f"DIFF_{key}_{diff_key}": diff_func(value) 
                   for key, value in cross_features.items() 
                   for diff_key, diff_func in diff_fx_axes.items() 
                   if max(map(int, diff_key[4:])) <= value.shape[1]}, axis=1)
            absdiffs = pd.concat({f"ABSDIFF_{key}_{diff_key}": diff_func(value.abs()) 
                   for key, value in cross_features.items() 
                   for diff_key, diff_func in diff_fx_axes.items() 
                   if max(map(int, diff_key[4:])) <= value.shape[1]}, axis=1)

            # Concatenate all features
            result = pd.concat((pd.concat(
                [code_out, mot_features, freq_features], axis=1).reset_index(drop=True), pd.concat([sensor_sums, sensor_abssums, sensor_corrs, sensor_abscorrs, sensor_diffs,
                 sensor_absdiffs, sums, abssums, corrs, abscorrs, diffs, absdiffs], axis=1).reset_index(drop=True)), axis=1)
            return result
        else:
            return pd.concat([code_out, freq_features], axis=1)


def find_neighbour(value, df, colname):
    exact_match = df[df[colname] == value]
    if not exact_match.empty:
        return exact_match.index[0]
    else:
        lower_neighbour_idx = df[df[colname] < value][colname].idxmax()
        upper_neighbour_idx = df[df[colname] > value][colname].idxmin()
        low_diff = abs(value - df[colname][lower_neighbour_idx])
        up_diff = abs(value - df[colname][upper_neighbour_idx])
        return lower_neighbour_idx if low_diff < up_diff else upper_neighbour_idx



def ratio_checking_function(df, column_name, min_ratio):
# Used to check whether the ratio of rows with most common position to all rows is high enough
    total_count = df.shape[0]
    if total_count == 0:
        return False
    position_counts = df[column_name].value_counts()
    dominant_position_count = position_counts.max()
    dominant_position_ratio = dominant_position_count / total_count
    return dominant_position_ratio >= min_ratio


def create_training_set_mod(dataframe, window_length_in_secs, min_samples, min_ratio_of_dominant_position=1):
    df = pd.DataFrame()
    grouped = dataframe.groupby('time_point') # Used to calculate features for 4 time_points simultaneously, to speed up the process
    group1 = grouped.get_group(4) # Features calculated for time_point 4 only in this specific example
    grouped2 = group1.groupby(['baby', 'study']) # Features calculated for each unique combination of 'baby', 'study' and 'time_point' values
    for name, group in grouped2:
      #for i in np.arange(0, (int(max(group['times [s]']) - min(group['times [s]'])) - window_length_in_secs), 0.5): # used for 1 second long windows, when step is not 1
      for i in range(int(max(group['times [s]']) - min(group['times [s]'])) - window_length_in_secs):
          start_index = find_neighbour(i, group, 'times [s]') # Since some rows from raw data might have been dropped, becuase of near zero variance, exact matches for integers (i) might not be present
          end_index = find_neighbour(i + window_length_in_secs, group, 'times [s]')
          #if group['baby'][start_index] == group['baby'][end_index]: # when desidered ratio = 1
          if ratio_checking_function(group.loc[start_index:end_index], 'position', min_ratio_of_dominant_position):
              df_slice = group.iloc[start_index - group.index.min():end_index - group.index.min() + 1]
              if df_slice.shape[0] > min_samples: # due to some dropped data, some windows might not contain enough samples
                  df = pd.concat((df, motion_features_mod(df_slice, int(start_index - group.index.min())/60, int(end_index - group.index.min())/60, True, ['sitting', 'hands_and_knees', 'crawling', 'supported_sitting_by_hand',
 'supported_stand', 'pull_to_stand', 'supine', 'prone', 'side_lying', 'supported_sitting_by_caregiver', 'held', 'standing_upright', 'reclined',
 'walking', 'supported_walking', 'belly_crawling', 'pivoting', 'sitting_rattling', 'crawling_rattling', 'supported_sitting_by_hand_rattling',
 'supported_stand_rattling', 'supported_sitting_by_caregiver_rattling', 'standing_upright_rattling', 'hands_and_knees_rattling', 'reclined_rattling',
 'walking_rattling', 'held_rattling', 'prone_rattling', 'belly_crawling_rattling', 'side_lying_rattling', 'supine_rattling', 'pivoting_rattling',
 'hand_supported_sitting', 'cg_supported_sitting', 'squatting', 'kneeling', 'kneeling-like', 'supported_kneeling'])), axis=0, ignore_index=True) # All of the position names used by human classifiers
    return df

data_file_path = os.path.join(os.pardir, 'data', 'whole_dataset.csv')
df = pd.read_csv(data_file_path)
set = create_training_set_mod(df, 2, 120, 0.75)
result_path = os.path.join(os.pardir, 'data', 'training_set_2s_75_tp4.csv')
set.to_csv(result_path, index=False)
