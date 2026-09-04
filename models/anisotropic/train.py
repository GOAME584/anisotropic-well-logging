"""
train.py
示例训练脚本（CSV 输入）。请根据实际列名和目标修改。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import shap
import json
import argparse

def make_features(df, feature_cols, window=5):
    X = df[feature_cols].copy()
    # 滑动统计（按深度顺序假设已排序）
    for col in feature_cols:
        X[f"{col}_mean_{window}"] = df[col].rolling(window, center=True, min_periods=1).mean()
        X[f"{col}_std_{window}"] = df[col].rolling(window, center=True, min_periods=1).std().fillna(0)
        X[f"{col}_grad"] = df[col].diff().fillna(0)
    # 填充缺失并返回
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return X

def main(args):
    np.random.seed(args.seed)

    df = pd.read_csv(args.input_csv)
    # 必须包含 'well_id' 分组列用于分组 CV；如果没有，可以用井名/文件名生成
    if 'well_id' not in df.columns:
        df['well_id'] = df.get('well', 'well_0')

    # 指定特征与目标
    feature_cols = args.features.split(",")
    target_col = args.target

    # 创建特征
    X = make_features(df, feature_cols, window=args.window)
    y = df[target_col].values
    groups = df['well_id'].values

    # 按井分割训练/测试（示例简单划分）
    well_ids = np.unique(groups)
    train_wells, test_wells = train_test_split(well_ids, test_size=0.2, random_state=args.seed)
    train_idx = np.isin(groups, train_wells)
    test_idx = np.isin(groups, test_wells)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 标准化（保存 scaler）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为 DMatrix 或使用 sklearn wrapper
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=1
    )

    eval_set = [(X_test_scaled, y_test)]
    model.fit(X_train_scaled, y_train,
              eval_set=eval_set,
              early_stopping_rounds=args.early_stopping,
              verbose=True)

    # 评估
    preds = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 保存模型与 scaler 与 feature list
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "xgb_model.json")
    model.save_model(model_path)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({"feature_cols": X.columns.tolist(), "target": target_col}, f, indent=2)

    # SHAP 分析（可选，数据量大则采样）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled[:1000])
    shap.summary_plot(shap_values, X_test.iloc[:1000], show=False)  # 可保存为图像

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--features", type=str, default="GR,RHOB,NPHI,DT,RES")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    main(args)
