import os
import ast
import json
import joblib
import warnings
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone 
from sklearn.model_selection import ParameterGrid, StratifiedKFold 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

warnings.filterwarnings('ignore')

def parse_bow_to_array(dataframe):
    bow_column = dataframe["bow"]
    parsed_lists = [
        [float(x) for x in ast.literal_eval(item)] if isinstance(item, str) else item
        for item in bow_column
    ]
    return np.array(parsed_lists)

def extract_bert_features(dataframe):
    return dataframe['bert'].apply(
        lambda feature: np.fromstring(feature[1:-1], sep=' ') if isinstance(feature, str) else feature
    )

def read_target_to_number_list(df):
    target_array = np.array(df[['target'][0]])
    return target_array

def read_feature_representation(df, feature_representation):
    feature_mapping = {
        "bow": parse_bow_to_array,
        "bert": extract_bert_features
    }

    if feature_representation in feature_mapping:
        X = feature_mapping[feature_representation](df)
        X = np.vstack(X)
    else:
        return None

    return X

def fetch_train_test_data(data, feature_representation, fraction=1):
    df = pd.read_csv(data)
    if 0 < fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)
    elif fraction != 1.0:
        return None
    target_array = read_target_to_number_list(df)
    feature_list = read_feature_representation(df, feature_representation)
    if feature_list is None:
        return None
    X_train, X_test, y_train, y_test = train_test_split(feature_list, target_array, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    return train_accuracy, test_accuracy

def train_and_evaluate(data, feature_representation, prediction_model, params={}, save_model=True):
    X_train, X_test, y_train, y_test = fetch_train_test_data(data, feature_representation)
    model = None
    if prediction_model == "logisticRegression":
        model = LogisticRegression(**params)
    elif prediction_model == "svc":
        model = SVC(**params)
    elif prediction_model == "naiveBayes":
        model = GaussianNB(**params)
    else:
        return None
    train_accuracy, test_accuracy = train_model(model, X_train, y_train, X_test, y_test)
    if save_model:
        params_str = "_".join(f"{key}_{value}" for key, value in params.items())
        model_name = f"{feature_representation}_{prediction_model}_{params_str}.joblib"
        joblib.dump(model, model_name)

def plot_pca_scatter(X_pca, y, labels, dimensions):
    for label in labels:
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f"Class {label}")
    plt.title(f"{dimensions}D PCA Scatter Plot")
    plt.legend()

def create_pca_visualization(data, feature_representation, dimensions):
    df = pd.read_csv(data)
    y = read_target_to_number_list(df)
    X = read_feature_representation(df, feature_representation)
    if X is None:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=dimensions)
    X_pca = pca.fit_transform(X_scaled)
    if dimensions == 2:
        unique_labels = np.unique(y)
        plot_pca_scatter(X_pca, y, unique_labels, dimensions)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(y)
        for label in unique_labels:
            ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1], X_pca[y == label, 2], label=label)
        plt.legend()
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
    else:
        return
    plt.savefig(f"{dimensions}D_PCA_{feature_representation}.png")
    plt.show()

def get_xd_lda_plot(data, feature_representation, dimensions):
    df = pd.read_csv(data)
    y = read_target_to_number_list(df)

    X = read_feature_representation(df, feature_representation)
    if X is None:
        return None
    

    lda = LDA(n_components=dimensions-1)
    X_lda = lda.fit_transform(X, y)

    if dimensions == 2:
        if 'y' in locals():
            unique_labels = np.unique(y)
            for label in unique_labels:
                plt.scatter(X_lda[y == label, 0], np.zeros_like(X_lda[y == label, 0]), label=label) 
            plt.legend()
        else:
            plt.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0])) 
        plt.xlabel('Linear Discriminant 1')
    else:
        print(f"{dimensions}D plotting not supported for LDA.")
        return

    plt.title(f"{dimensions}D LDA of the Dataset for {feature_representation} Feature Representation")

    filename = f"{dimensions}D_LDA_{feature_representation}.png"
    plt.savefig(filename) 
    print(f"Figure saved as {filename}")

    plt.show() 

def scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return accuracy_score(y, predictions)

def fit_and_score_estimator(progress_bar, estimator, X, y, train, test, scorer, parameters):
    est = clone(estimator)
    est.set_params(**parameters)
    est.fit(X[train], y[train])
    train_score = scorer(est, X[train], y[train])
    test_score = scorer(est, X[test], y[test])
    progress_bar.update(1) 
    return {'params': parameters, 'train_score': train_score, 'test_score': test_score}

def tune_and_report(data, feature_representation, prediction_model, frac=1):
    X_train, X_test, y_train, y_test = fetch_train_test_data(data, feature_representation, frac)
    param_grid = None
    model = None
    if prediction_model == "logisticRegression":
        param_grid = [
            {'penalty': ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},
            {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
            {'penalty': ['elasticnet'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'l1_ratio': [0, 0.5, 1]},
            {'penalty': [None], 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
        ]
        model = LogisticRegression(max_iter=5000)
    elif prediction_model == "svc":
        param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'degree': [2, 3, 4],
             'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
             'tol': [1e-3, 1e-4],
             'class_weight': [None, 'balanced']
            }
        ]
        model = SVC(max_iter=5000)
    elif prediction_model == "naiveBayes":
        param_grid = [{'var_smoothing': np.logspace(0, -9, num=100)}]
        model = GaussianNB()
    else:
        return None
    total_combinations = sum(len(list(product(*grid.values()))) for grid in param_grid) * 5
    progress_bar = tqdm(total=total_combinations)
    results = joblib.Parallel(n_jobs=-1, backend="threading")(joblib.delayed(
        fit_and_score_estimator)(progress_bar, model, X_train, y_train, train_index, test_index, scorer, parameters)
        for parameters in ParameterGrid(param_grid)
        for train_index, test_index in StratifiedKFold(n_splits=5).split(X_train, y_train))
    progress_bar.close()
    with open(f"result_{frac}_{feature_representation}_{prediction_model}.json", 'w') as json_file:
        print(f"Saving results to result_{frac}_{feature_representation}_{prediction_model}.json")
        json.dump(results, json_file, indent=4)

def get_confusion_matrix(model_path, data):
    model = joblib.load(model_path)
    model_name = os.path.basename(model_path).replace('.joblib', '')
    X_train, X_test, y_train, y_test = fetch_train_test_data(data, model_path.split("_")[0])
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{model_name}_Confusion_Matrix.png')

def plot_scores_from_var_smoothing(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    params_df = pd.json_normalize(df['params'])
    df = pd.concat([df.drop(['params'], axis=1), params_df], axis=1)
    agg_df = df.groupby('var_smoothing').mean().reset_index()
    parts = file_path.split(".")[0].split("_")
    feature_representation, model_name = parts[-2], parts[-1]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(agg_df['var_smoothing'], agg_df['train_score'], marker='o', linestyle='-', color='blue')
    plt.title(f'Train Score vs Var Smoothing ({feature_representation}, {model_name})')
    plt.xlabel('Var Smoothing')
    plt.ylabel('Train Score')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(agg_df['var_smoothing'], agg_df['test_score'], marker='o', linestyle='-', color='green')
    plt.title(f'Evaluation Score vs Var Smoothing ({feature_representation}, {model_name})')
    plt.xlabel('Var Smoothing')
    plt.ylabel('Evaluation Score')
    plt.grid(True)
    plt.tight_layout()
    save_filename = f"{model_name}_{feature_representation}_train_and_evaluation_scores.png"
    plt.savefig(save_filename)

def plot_scores_by_C(file_path, specific_params):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data)
    df['train_score'] = pd.to_numeric(df['train_score'], errors='coerce')
    df['test_score'] = pd.to_numeric(df['test_score'], errors='coerce')
    feature_representation = file_path.split("_")[-2]
    model_name = file_path.split("_")[-1].split(".")[0]
    for param, value in specific_params.items():
        df = df[df[f'params.{param}'] == value]
    if df.empty:
        print("No data matches the specified parameters.")
        return
    df['params.C'] = pd.to_numeric(df['params.C'], errors='coerce')
    df.dropna(subset=['params.C', 'train_score', 'test_score'], inplace=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df['params.C'], df['train_score'], marker='o', linestyle='-', color='blue')
    plt.title(f'Train Score vs C ({feature_representation}, {model_name})')
    plt.xlabel('C')
    plt.ylabel('Train Score')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(df['params.C'], df['test_score'], marker='o', linestyle='-', color='green')
    plt.title(f'Evaluation Score vs C ({feature_representation}, {model_name})')
    plt.xlabel('C')
    plt.ylabel('Evaluation Score')
    plt.grid(True)
    plt.tight_layout()
    specific_params_str = '_'.join([f"{key}_{value}" for key, value in specific_params.items()]).replace('=', '').replace(',', '')
    filename = f"{model_name}_{feature_representation}_{specific_params_str}_scores.png"
    plt.savefig(filename)

def main():
    data = "embeddings_15000.csv"

    tune_and_report(data, "bow", "logisticRegression")
    plot_scores_by_C('result_1_bow_logisticRegression.json', {"penalty": "l2", "solver": "newton-cg"})

    tune_and_report(data, "bert", "logisticRegression")
    plot_scores_by_C('result_1_bert_logisticRegression.json', {"penalty": "l2", "solver": "newton-cg"})

    tune_and_report(data, "bert", "naiveBayes")
    plot_scores_from_var_smoothing('result_1_bert_naiveBayes.json')

    tune_and_report(data, "bow", "naiveBayes")
    plot_scores_from_var_smoothing('result_1_bow_naiveBayes.json')

    tune_and_report(data, "bow", "svc")
    plot_scores_by_C('result_0.5_bow_svc.json', {"degree": 2, "gamma": "scale", "kernel": "linear"})

    tune_and_report(data, "bert", "svc")
    plot_scores_by_C('result_0.5_bert_svc.json', {"degree": 4, "gamma": "scale", "kernel": "rbf"})

    train_and_evaluate(data, "bow", "logisticRegression")
    train_and_evaluate(data, "bert", "logisticRegression")
    train_and_evaluate(data, "bow", "svc")
    train_and_evaluate(data, "bert", "svc")
    train_and_evaluate(data, "bow", "naiveBayes")
    train_and_evaluate(data, "bert", "naiveBayes")
    create_pca_visualization(data, "bert", 2)
    create_pca_visualization(data, "bow", 2)
    create_pca_visualization(data, "bert", 3)
    create_pca_visualization(data, "bow", 3)
    get_xd_lda_plot(data, "bert", 2)
    get_xd_lda_plot(data, "bow", 2)
    
    current_directory = os.getcwd()
    files_in_directory = os.listdir(current_directory)
    joblib_files = [file for file in files_in_directory if file.endswith('.joblib')]
    for model_path in joblib_files:
        get_confusion_matrix(model_path, data)

if __name__ == "__main__":
    main()
