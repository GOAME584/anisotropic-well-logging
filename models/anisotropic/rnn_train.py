"""
rnn_train.py
用 LSTM 建模深度序列（按井做序列）。
每口井用滑动窗口构造样本：window_size 长度的多变量序列 -> 预测中心或最后位置的 target。
运行:
  python models/anisotropic/rnn_train.py --input_csv sample_wells.csv --output_dir models/rnn_model
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_sequences(df, feature_cols, window=21, stride=1, label_pos="center"):
    # df: 单井的 DataFrame（已按 depth 排序）
    arr = df[feature_cols].values
    n = len(arr)
    half = window // 2
    Xs, ys = [], []
    for i in range(0, n - window + 1, stride):
        seq = arr[i:i+window]
        if label_pos == "center":
            label_idx = i + half
        elif label_pos == "last":
            label_idx = i + window - 1
        else:
            label_idx = i + half
        Xs.append(seq)
        ys.append(df.iloc[label_idx]['target'])
    return np.array(Xs), np.array(ys)


def build_model(window, n_features, hidden_units=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=(window, n_features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=False)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mae'])
    return model


def main(args):
    np.random.seed(args.seed)

    df = pd.read_csv(args.input_csv)
    if 'well_id' not in df.columns:
        df['well_id'] = df.get('well', 'well_0')

    feature_cols = args.features.split(",")

    # 按井构造序列
    X_list, y_list = [], []
    for well, g in df.groupby('well_id'):
        g = g.sort_values('depth').reset_index(drop=True)
        Xs, ys = make_sequences(g, feature_cols, window=args.window, stride=args.stride, label_pos=args.label_pos)
        if len(Xs) > 0:
            X_list.append(Xs)
            y_list.append(ys)

    if len(X_list) == 0:
        raise RuntimeError("No sequences generated. Check window/stride or input formatting.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # 标准化特征（按通道）
    nsamples, w, nfeat = X.shape
    X_reshaped = X.reshape(-1, nfeat)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_reshaped).reshape(nsamples, w, nfeat)

    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, random_state=args.seed)

    model = build_model(args.window, len(feature_cols), hidden_units=args.hidden_units)

    callbacks = []
    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping, restore_best_weights=True))

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "rnn_model.h5"))
    import joblib
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    print("Saved RNN model to", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--features", default="GR,RHOB,NPHI,DT,RES")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--label_pos", choices=["center","last"], default="center")
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="models/rnn_model")
    args = parser.parse_args()
    main(args)
