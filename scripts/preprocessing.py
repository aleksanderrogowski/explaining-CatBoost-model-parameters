import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def apply_filter(data, b, a):
    return filtfilt(b, a, data)


def separate_ac_dc_components(df, cutoff=2, fs=60):
    b_low, a_low = butter_lowpass(cutoff, fs, order=1)
    b_high, a_high = butter_highpass(cutoff, fs, order=1)
    dc_components = pd.DataFrame()
    ac_components = pd.DataFrame()
    for col in df.columns:
        signal = df[col].values
        dc_component = apply_filter(signal, b_low, a_low)
        dc_components[col] = dc_component
        ac_component = apply_filter(signal, b_high, a_high)
        ac_components[col] = ac_component
    return dc_components, ac_components


def splitter(array, win_len, overlap):
    strides = np.arange(0, len(array) - win_len + 1, win_len - overlap)
    return np.array([array[i:i + win_len] for i in strides])


def variance(array, time_vector, win_len, overlap):
    fragments = splitter(array, win_len, overlap)
    variance_vector = np.zeros(time_vector.size)
    for n, fragment in enumerate(fragments):
        variance_vector[n * (win_len - overlap):n * (win_len - overlap) + win_len] = np.var(fragment)
    return variance_vector


def get_time_point(baby_name):
    return str(baby_name)[-1] if str(baby_name)[-1] in ('1', '2', '3', '4') else None


def delete_baby_string(baby_name):
    return str(baby_name)[:-5]


def col_avg(df, col_ids):
    means = df[col_ids].mean()
    df[col_ids] -= means
    return df

directory_path = os.path.join(os.pardir, 'example_data_files')
directory_files = os.listdir(directory_path)
events = [name for name in directory_files if 'events' in name]
data = [name for name in directory_files if 'data' in name]

events_df = pd.concat((pd.read_csv(os.path.join(directory_path, plik), delimiter="\t") for plik in events))
events_df['time_point'] = events_df['baby'].apply(get_time_point)
sensors = ['infanttrunkmid', 'infanttrunkleft', 'infantleftleg', 'infantrightleg']  # when using 4 of the 6 sensors
t = 0.1
Fs = 60

data_df = pd.DataFrame()
for item in data:
    df = pd.read_csv(os.path.join(directory_path, item), delimiter="\t")
    df = df.drop(columns=[col for col in df.columns if 'arm' in col])
    df['baby'] = item[:7]
    df['time_point'] = item[6:7]
    study = item.split('.')[0].split('_')[2]
    df['study'] = study
    df['times [s]'] = df['times'] / 1000
    acc_columns = [col for col in df.columns if 'Acc' in col]
    # Acc signal processing into AC, DC and raw signal
    for column in acc_columns:
        dc_components, ac_components = separate_ac_dc_components(pd.DataFrame(df[column]), cutoff=2, fs=60)
        df['DC_' + column] = dc_components
        df['AC_' + column] = ac_components
    df['position'] = None

    for sensor in sensors:
        acc_x = df['Acc_X_' + sensor]
        acc_y = df['Acc_Y_' + sensor]
        acc_z = df['Acc_Z_' + sensor]
        no_nan = ~(acc_x.isnull() | acc_y.isnull() | acc_z.isnull())
        df = df[no_nan]
        if df.shape[0] == 0: continue
        module = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        roll = np.arctan2(acc_y, acc_z)
        pitch = np.arctan2(-acc_x, np.sqrt(acc_y ** 2 + acc_z ** 2))
        df['Acc_module_' + sensor] = module
        dc_acc, ac_acc = separate_ac_dc_components(pd.DataFrame(df['Acc_module_' + sensor]), cutoff=2, fs=60)
        df['DC_Acc_module_' + sensor] = dc_acc
        df['AC_Acc_module_' + sensor] = ac_acc
        df['roll_' + sensor] = roll
        df['pitch_' + sensor] = pitch
        df['Acc_variance_' + sensor] = variance(module, df['times [s]'], int(t * Fs), 0)
        df = df[df['Acc_variance_' + sensor] >= 1e-6]
        df = df.drop(columns=['Acc_variance_' + sensor])
    df = df.drop(columns=['times', 'ParenthandL', 'ParenthandR'])
    for _, row in events_df.iterrows():
        if study == 'books':
            condition = (df['baby'] == row['baby']) & \
                    (df['times [s]'] >= row['latency']/1000) & \
                    (df['times [s]'] <= row['latency']/1000 + row['duration']/1000)
        elif study != 'books':
            condition = (df['baby'] == row['baby']) & \
                        (df['times [s]'] >= row['latency']) & \
                        (df['times [s]'] <= row['latency'] + row['duration'])
        df.loc[condition, 'position'] = row['type']
    data_df = pd.concat((data_df, df))

# Save the resulting dataset
data_folder_path = os.path.join(os.pardir, 'data')
os.makedirs(data_folder_path, exist_ok=True)
file_path = os.path.join(data_folder_path, 'whole_dataset.csv')
data_df.to_csv(file_path, index=False)
