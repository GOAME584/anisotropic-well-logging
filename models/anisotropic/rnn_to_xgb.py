"""
rnn_to_xgb.py
流程：
1) 用 RNN（同 rnn_train 中的网络）训练或加载网络。
2) 把每个窗口的中间表示（例如 LSTM 的 hidden state）作为 embedding（特征）。
3) 把 embedding 附加到现有特征后训练 XGBoost/RF。
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def make_sequences(df, feature_cols, window=21, label_pos="center"):
    arr = df[feature_cols].values
    n = len(arr)
    half = window // 2
    Xs, ys, idxs = [], [], []
    for i in range(0, n - window + 1):
        seq = arr[i:i+window]
        if label_pos=="center":
            label_idx = i + half
        else:
            label_idx = i + window - 1
        Xs.append(seq)
        ys.append(df.iloc[label_idx]['target'])
        idxs.append(df.index[label_idx])
    return np.array(Xs), np.array(ys), np.array(idxs)


def extract_embeddings(model, X_seq, batch_size=512):
    # 假设 model 的 penultimate 层是我们要的 embedding
    intermediate_layer = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    return intermediate_layer.predict(X_seq, batch_size=batch_size)


def main(args):
    df = pd.read_csv(args.input_csv)
    if 'well_id' not in df.columns:
        df['well_id'] = df.get('well', 'well_0')

    feature_cols = args.features.split(",")
    Xs_all, ys_all, idxs_all = [], [], []
    for well, g in df.groupby('well_id'):
        g = g.sort_values('depth').reset_index(drop=True)
        Xs, ys, idxs = make_sequences(g, feature_cols, window=args.window, label_pos=args.label_pos)
        if len(Xs)>0:
            Xs_all.append(Xs); ys_all.append(ys); idxs_all.append(idxs)
    if len(Xs_all)==0:
        raise RuntimeError("No sequences generated.")
    X_seq = np.concatenate(Xs_all, axis=0)
    y = np.concatenate(ys_all, axis=0)

    # 标准化
    nsamples, w, nfeat = X_seq.shape
    X_flat = X_seq.reshape(-1, nfeat)
    scaler = StandardScaler()
    Xs_scaled = scaler.fit_transform(X_flat).reshape(nsamples, w, nfeat)

    # 加载已训练的 RNN 或退出
    if os.path.exists(args.rnn_model):
        rnn = tf.keras.models.load_model(args.rnn_model)
    else:
        raise RuntimeError("Please train an RNN first or provide --rnn_model")

    embeddings = extract_embeddings(rnn, Xs_scaled, batch_size=args.batch_size)
    # embeddings 形状 (nsamples, emb_dim)

    # 为每个样本构造传统静态特征（例如窗口中心的静态特征）
    center_idx = args.window//2
    static_feats = Xs_scaled[:, center_idx, :]

    # 合并 embedding 与 static features
    X_final = np.hstack([static_feats, embeddings])

    # 训练 XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=args.seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'objective':'reg:squarederror', 'eta':args.learning_rate, 'max_depth':args.max_depth, 'subsample':args.subsample}
    bst = xgb.train(params, dtrain, num_boost_round=args.num_boost_round, evals=[(dtest,'eval')], early_stopping_rounds=args.early_stopping)
    bst.save_model(args.xgb_out)
    joblib.dump(scaler, args.scaler_out)
    print("Saved XGBoost model:", args.xgb_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--features", default="GR,RHOB,NPHI,DT,RES")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--label_pos", choices=["center","last"], default="center")
    parser.add_argument("--rnn_model", default="models/rnn_model/rnn_model.h5")
    parser.add_argument("--xgb_out", default="models/xgb_from_rnn.json")
    parser.add_argument("--scaler_out", default="models/rnn_scaler.joblib")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_boost_round", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
