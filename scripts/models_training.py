import joblib
import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


# Function to remove the last two characters from the 'baby' column
def delete_time_point(baby_name):
    return str(baby_name)[:-2]


# Function to compute class weights for imbalanced datasets
def compute_class_weights(y):
    class_weights = {cls: len(y) / (len(np.unique(y)) * list(y).count(cls)) for cls in np.unique(y)}
    return class_weights


# Function to determine feature type based on its name
def get_feature_type(feature_name):
    if any(summary in feature_name for summary in ['MEAN', 'SD', 'MED', 'SKEW', 'KURT', 'MIN', 'MAX', 'P25', 'P75']):
        return 'Statistical'
    elif any(feature in feature_name for feature in ['_energy', '_entropy', '_centroid', '_bandwidth', '_max_freq']):
        return 'Frequency'
    elif 'MAG' in feature_name or 'SUM' in feature_name:
        return 'Summary'
    elif 'DIFF' in feature_name:
        return 'Difference'
    elif 'CORR' in feature_name:
        return 'Correlation'
    return 'Other'


# Function to train and save models
def train_and_save_models(X_train, y_train, X_test, y_test, fold_idx, feature_group_name, model_path):
    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        class_weights=class_weights,
        custom_metric=['Accuracy'],
        verbose=100
    )

    # Train model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)

    # Evaluate and print results (optional, mainly for debugging)
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"Classification report for fold {fold_idx + 1}, feature group {feature_group_name}:")
    print(clf_report)
    print(f"Confusion matrix for fold {fold_idx + 1}, feature group {feature_group_name}:")
    print(cm)


# Load and preprocess the dataset
data_file_path = os.path.join(os.pardir, 'data', 'training_set_2s_75_tp4.csv')
df = pd.read_csv(data_file_path)
df['baby'] = df['baby'].apply(delete_time_point)

position_mapping = {
    'sitting': 'sitting', 'hands_and_knees': 'hands_and_knees', 'crawling': 'hands_and_knees',
    'supported_sitting_by_hand': 'sitting', 'supported_stand': 'upright', 'side_lying': 'prone',
    'supported_sitting_by_caregiver': 'sitting', 'prone': 'prone',
    'walking': 'upright', 'supine': 'supine', 'standing_upright': 'upright',
    'supported_walking': 'upright', 'belly_crawling': 'prone', 'pivoting': 'prone',
    'cg_supported_sitting': 'sitting', 'hand_supported_sitting': 'sitting'
}

df['mapped_position'] = df['position'].map(position_mapping)
unmapped_rows = df['mapped_position'].isna()
train_df = df[~unmapped_rows]

# Feature and target variables
X = train_df.drop(columns=['baby', 'start', 'end', 'position', 'mapped_position', 'study', 'time_point'])
y = train_df['mapped_position']
groups = train_df['baby']

# Parameters for model training
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=163)

# Automatically generate feature groups based on feature names
feature_groups = {}
for col in X.columns:
    feature_type = get_feature_type(col)
    if feature_type not in feature_groups:
        feature_groups[feature_type] = []
    feature_groups[feature_type].append(col)

# Unique classes from the mapping
classes = sorted(set(position_mapping.values()))

# Ensure the directory exists
base_dir = os.path.join(os.pardir, 'models')
os.makedirs(base_dir, exist_ok=True)

# Iterate over each fold
for fold_idx, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train model using all features
    model_path = os.path.join(base_dir, f'model_2s_75_no_arms_all_features_fold_{fold_idx + 1}.sav')
    train_and_save_models(
        X_train, y_train, X_test, y_test, fold_idx,
        feature_group_name="All Features",
        model_path=model_path
    )

    # Train models for each feature group
    for feature_group_name, feature_group_cols in feature_groups.items():
        X_train_group = X_train[feature_group_cols]
        X_test_group = X_test[feature_group_cols]

        model_path = os.path.join(base_dir, f'model_2s_75_no_arms_{feature_group_name}_fold_{fold_idx + 1}.sav')
        train_and_save_models(
            X_train_group, y_train, X_test_group, y_test, fold_idx,
            feature_group_name=feature_group_name,
            model_path=model_path
        )
