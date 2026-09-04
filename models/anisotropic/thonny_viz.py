"""
thonny_viz.py
Thonny 友好的可视化脚本：绘制深度曲线、预测 vs 真实、方位玫瑰图
运行:
  python thonny_viz.py --input-csv well_sample.csv --model models/xgb.json
在 Thonny 中直接打开并运行，会弹出 matplotlib 图窗或保存 PNG。
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import json
import os

def load_model(model_path, meta_path=None):
    # 先尝试 xgboost Booster json
    booster = xgb.Booster()
    booster.load_model(model_path)
    meta = None
    if meta_path and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    return booster, meta

def plot_logs_and_prediction(df, preds, feature_cols, target_col='target', out_prefix=None):
    depth = df['depth'].values
    ncols = len(feature_cols) + 1  # features + prediction
    fig, axes = plt.subplots(1, ncols, figsize=(3*ncols, 8), sharey=True)
    for i, col in enumerate(feature_cols):
        axes[i].plot(df[col], depth, label=col)
        axes[i].invert_yaxis()
        axes[i].set_xlabel(col)
        axes[i].grid(True)
    # prediction plot
    axes[-1].plot(preds, depth, label='pred', color='red')
    if target_col in df.columns:
        axes[-1].plot(df[target_col], depth, label='true', color='black', alpha=0.7)
    axes[-1].set_xlabel('prediction / true')
    axes[-1].legend()
    plt.suptitle('Well Logs and Prediction')
    plt.tight_layout()
    if out_prefix:
        plt.savefig(out_prefix + "_logs_pred.png", dpi=150)
    plt.show()

def plot_azimuth_rose(df, az_col='azimuth', value_col=None, n_bins=36, out_prefix=None):
    # 画方位玫瑰图：按角度柱状
    az = df[az_col].dropna().values % 360
    bins = np.linspace(0, 360, n_bins+1)
    counts, _ = np.histogram(az, bins=bins)
    theta = np.deg2rad((bins[:-1] + bins[1:]) / 2)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    # bar width
    width = 2*np.pi / n_bins
    bars = ax.bar(theta, counts, width=width, bottom=0.0, color='C0', alpha=0.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.title('Azimuth Rose')
    if out_prefix:
        plt.savefig(out_prefix + "_azimuth_rose.png", dpi=150)
    plt.show()

def main(args):
    df = pd.read_csv(args.input_csv)
    # 加载模型元数据获得特征列
    meta = {}
    if os.path.exists(args.meta):
        meta = json.load(open(args.meta))
    if 'feature_cols' in meta:
        feature_cols = meta['feature_cols'][:5]  # 这里只画前几个主要特征
    else:
        # 简单选择一些列
        feature_cols = [c for c in ['GR','RHOB','NPHI','DT','RES'] if c in df.columns]

    # 尝试用模型预测（Booster via DMatrix）
    try:
        booster, _ = load_model(args.model, args.meta)
        # 构造 X 与特征顺序（若 meta 提供 feature_cols 则使用）
        if 'feature_cols' in meta:
            X = df[meta['feature_cols']].fillna(0).values
            dmat = xgb.DMatrix(X, feature_names=meta['feature_cols'])
            preds = booster.predict(dmat)
        else:
            # 简化：用 available feature_cols
            X = df[feature_cols].fillna(0).values
            dmat = xgb.DMatrix(X, feature_names=feature_cols)
            preds = booster.predict(dmat)
    except Exception as e:
        print("Model load/predict failed:", e)
        preds = np.zeros(len(df))

    # 可视化
    out_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.input_csv))[0]
    plot_logs_and_prediction(df, preds, feature_cols, target_col=args.target, out_prefix=out_prefix)
    if 'azimuth' in df.columns:
        plot_azimuth_rose(df, az_col='azimuth', out_prefix=out_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/xgb.json")
    parser.add_argument("--meta", type=str, default="models/meta.json")
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--target", type=str, default="target")
    args = parser.parse_args()
    main(args)
