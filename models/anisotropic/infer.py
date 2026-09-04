"""
infer.py
示例推理脚本：加载模型与 scaler，预测新井的目标属性（CSV 输入）。
"""
import argparse
import pandas as pd
import joblib
import xgboost as xgb
import json
import numpy as np
from train import make_features  # 假设同目录

def load_model(model_dir):
    model = xgb.Booster()
    model_path = f"{model_dir}/xgb_model.json"
    model.load_model(model_path)
    # 如果用 XGBRegressor 保存的 JSON，需要使用 XGBRegressor().load_model 或 joblib
    return model

def main(args):
    df = pd.read_csv(args.input_csv)
    meta = json.load(open(f"{args.model_dir}/meta.json"))
    feature_cols = meta['feature_cols']
    scaler = joblib.load(f"{args.model_dir}/scaler.joblib")

    X = make_features(df, feature_cols=[c.split("_")[0] for c in feature_cols if "_" in c or c in df.columns], window=args.window)
    # 对齐列顺序
    X = X.reindex(columns=[c for c in feature_cols], fill_value=0)
    X_scaled = scaler.transform(X)

    # 加载 sklearn-wrapper 模型如果存在
    try:
        model = joblib.load(f"{args.model_dir}/xgb_sklearn_model.joblib")
        preds = model.predict(X_scaled)
    except Exception:
        # 作为后备，加载 Booster 并使用 DMatrix
        model = xgb.Booster()
        model.load_model(f"{args.model_dir}/xgb_model.json")
        dmat = xgb.DMatrix(X_scaled, feature_names=X.columns)
        preds = model.predict(dmat)

    df['prediction'] = preds
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_csv", type=str, default="preds.csv")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    main(args)
