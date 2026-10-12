"""
benchmark_models.py
====================
Benchmarks 9 regression models (tree-based and scale-sensitive) against a
labeled CFST dataset and visualizes relative accuracy on a Taylor Diagram
(correlation, standard deviation, and RMSE in one polar plot).

Expects a tabular dataset (.xlsx or .csv) where the LAST column is the
regression target and every other column is a feature.

NOTE: this is a separate benchmarking workflow from generate_data.py /
train_lstm.py -- it compares classical ML models on a labeled dataset
(e.g. 'processed_data2.xlsx'), rather than training the project's own
dual-output LSTM on synthetic physics data. The source notebook
(notebooks/model_comparison.ipynb) additionally includes an extended
metrics/radar-chart comparison and a 3D hyperparameter grid-search
exploration; those are kept in the notebook rather than duplicated here.

Usage:
    python src/benchmark_models.py --data processed_data2.xlsx
    python src/benchmark_models.py --data results.csv --out-dir results
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


def load_dataset(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Loaded '{path}': {df.shape[0]} rows, {df.shape[1]} columns (target: '{target_col}')")
    return X, y


def build_models():
    """Scale-sensitive and tree-based model groups, trained differently below."""
    models_needing_scaling = {
        "LR": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVM": SVR(C=100, epsilon=0.1),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }
    models_tree_based = {
        "DT": DecisionTreeRegressor(max_depth=10, random_state=42),
        "RF": RandomForestRegressor(n_estimators=100, random_state=42),
        "AB": AdaBoostRegressor(n_estimators=50, random_state=42),
        "LB": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "XB": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }
    return models_needing_scaling, models_tree_based


def train_and_predict(X_train, X_test, y_train):
    scale_models, tree_models = build_models()

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    predictions = {}
    print("\n--- Training models ---")
    for name, model in scale_models.items():
        model.fit(X_train_sc, y_train)
        predictions[name] = model.predict(X_test_sc)
        print(f"  {name} trained (scaled features)")

    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        print(f"  {name} trained (raw features)")

    return predictions


def plot_taylor_diagram(ref, preds_dict, out_path=None, show_plot=True):
    ref_std = np.std(ref)

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.plot([0], [ref_std], "k*", markersize=14, label="Reference")

    max_std = ref_std
    for p in preds_dict.values():
        max_std = max(max_std, np.std(p))
    ax.set_rlim(0, max_std * 1.1)

    r_labels = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.0]
    t_locs = np.arccos(r_labels)
    ax.set_thetagrids(np.degrees(t_locs), labels=r_labels)
    ax.set_ylabel("Standard Deviation", labelpad=25)
    ax.text(np.radians(45), max_std * 1.15, "Correlation Coefficient", rotation=-45, ha="center")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    markers = ["o", "s", "D", "^", "v", "p", "X", "h", "*"]

    print(f"\n{'Model':<10} | {'RMSE':<10} | {'Corr':<10} | {'Std Dev':<10}")
    print("-" * 45)

    for i, (name, pred) in enumerate(preds_dict.items()):
        std = np.std(pred)
        corr = np.corrcoef(ref, pred)[0, 1]
        rmse = np.sqrt(mean_squared_error(ref, pred))

        theta = np.arccos(corr)
        r = std
        ax.plot(theta, r, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                linestyle="None", label=name, markersize=10, markeredgecolor="black")
        print(f"{name:<10} | {rmse:<10.2f} | {corr:<10.3f} | {std:<10.2f}")

    ax.set_title("Taylor Diagram: All Models", y=1.05, fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved Taylor diagram to '{out_path}'")
    if show_plot:
        plt.show()


def run(data_path, out_dir="results", show_plot=True):
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictions = train_and_predict(X_train, X_test, y_train)
    plot_taylor_diagram(
        y_test, predictions,
        out_path=os.path.join(out_dir, "taylor_diagram.png"),
        show_plot=show_plot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True, help="Path to labeled dataset (.xlsx or .csv), target in the last column")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to write the Taylor diagram to")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a display window")
    args = parser.parse_args()

    run(data_path=args.data, out_dir=args.out_dir, show_plot=not args.no_show)
