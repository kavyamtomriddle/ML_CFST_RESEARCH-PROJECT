"""
train_lstm.py
=============
Trains the "Glass-Box" Dual-Output Seq2Seq LSTM surrogate model.

Maps 4 static design parameters [D, t, fy, fc] to a 100-step time series
[M(phi), c(phi)] -- jointly predicting the Moment capacity AND the internal
Neutral Axis depth, which exposes the model's internal physical state rather
than treating it as a black box.

Expects a dataset produced by generate_data.py:
    data/cfst_dataset.npz  (X_train, X_test, y_train, y_test)

Output:
    models/cfst_glass_box_model.h5
    models/glass_box_verification.png   (true vs. predicted, both channels)

Usage:
    python src/train_lstm.py
    python src/train_lstm.py --data data/cfst_dataset.npz --epochs 150
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential

SEQ_LENGTH = 100
INPUT_DIM = 4    # D, t, fy, fc
OUTPUT_DIM = 2   # [Moment, Neutral Axis]


def build_glass_box_model():
    model = Sequential([
        Input(shape=(INPUT_DIM,)),
        RepeatVector(SEQ_LENGTH),

        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.1),

        LSTM(64, return_sequences=True, activation="tanh"),
        Dropout(0.1),

        # Predicts [Moment, Neutral_Axis] at every step -- this is what
        # makes the model a "glass box": its internal physical state is
        # directly observable, not hidden.
        TimeDistributed(Dense(OUTPUT_DIM, activation="linear")),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run(data_path="data/cfst_dataset.npz", out_dir="models", epochs=150, batch_size=64, show_plot=False):
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    print("--- DATA READY ---")
    print(f"Input: {INPUT_DIM} static features")
    print(f"Output: {SEQ_LENGTH} steps x {OUTPUT_DIM} variables (glass-box mode)")
    print("y_train shape:", y_train.shape, "(expected: (N, 100, 2))")

    model = build_glass_box_model()
    model.summary()

    print("\n--- STARTING TRAINING ---")
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    print("\n--- EVALUATING ON HELD-OUT TEST SET ---")
    preds = model.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    indices = np.random.choice(len(X_test), 3, replace=False)
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    for i, idx in enumerate(indices):
        ax1.plot(y_test[idx, :, 0], color=colors[i], linewidth=2.5, alpha=0.6, label=f"True Sample {idx}")
        ax1.plot(preds[idx, :, 0], color=colors[i], linestyle="--", linewidth=2, label=f"AI Sample {idx}")
    ax1.set_title("Target 1: Flexural Capacity (Moment)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Strain Step")
    ax1.set_ylabel("Normalized Moment")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for i, idx in enumerate(indices):
        ax2.plot(y_test[idx, :, 1], color=colors[i], linewidth=2.5, alpha=0.6, label=f"True Sample {idx}")
        ax2.plot(preds[idx, :, 1], color=colors[i], linestyle="--", linewidth=2, label=f"AI Sample {idx}")
    ax2.set_title("Target 2: Internal Equilibrium (Neutral Axis)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Strain Step")
    ax2.set_ylabel("Normalized NA Depth")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "glass_box_verification.png")
    plt.savefig(fig_path, dpi=150)
    if show_plot:
        plt.show()
    print(f"Saved verification figure to '{fig_path}'")

    model_path = os.path.join(out_dir, "cfst_glass_box_model.h5")
    model.save(model_path)
    print(f"Model saved to '{model_path}'")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/cfst_dataset.npz", help="Path to dataset produced by generate_data.py")
    parser.add_argument("--out-dir", type=str, default="models", help="Directory to write the trained model + figure to")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot", action="store_true", help="Show the verification plot interactively")
    args = parser.parse_args()

    run(
        data_path=args.data,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        show_plot=args.plot,
    )
