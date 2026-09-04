"""
big_data_pipeline.py
Dask-based 大数据特征工程示例（针对各向异性特征）
运行示例（单机 dask）:
  from dask.distributed import Client
  client = Client(n_workers=4)
  python big_data_pipeline.py --input-parquet /data/logs/ --output-parquet /tmp/feats/
"""
import argparse
import dask.dataframe as dd
import numpy as np
import pandas as pd

def compute_directional_slices(df, azimuth_col='azimuth', value_cols=None, n_slices=8):
    """
    为每个深度点按 azimuth 切分周向统计（近似：如果我们有测井环向采样或相邻井可做）
    这里示例：将 azimuth 分桶后计算每桶的均值（在同一分区内）
    """
    if value_cols is None:
        value_cols = ['GR','RHOB','NPHI']
    df = df.copy()
    # 归一化 azimuth 到 0-360
    df[azimuth_col] = df[azimuth_col] % 360
    bins = np.linspace(0, 360, n_slices+1)
    df['az_bin'] = pd.cut(df[azimuth_col], bins=bins, labels=False, include_lowest=True)
    agg = df.groupby(['well_id','depth','az_bin'])[value_cols].mean().unstack(fill_value=0)
    # 展平列名
    agg.columns = [f"{val}_az{int(b)}" for val,b in agg.columns]
    agg = agg.reset_index()
    # 合并回原表（按 well_id, depth）
    merged = pd.merge(df, agg, on=['well_id','depth'], how='left')
    return merged

def make_features_partition(pdf, feature_cols, window=11, n_slices=8):
    # pdf 是 pandas DataFrame（分区）
    pdf = pdf.sort_values('depth')
    X = pdf[feature_cols].copy()
    for col in feature_cols:
        X[f"{col}_mean_{window}"] = pdf[col].rolling(window, center=True, min_periods=1).mean()
        X[f"{col}_std_{window}"] = pdf[col].rolling(window, center=True, min_periods=1).std().fillna(0)
        X[f"{col}_grad"] = pdf[col].diff().fillna(0)
    # 添加方向切片统计（在分区范围内）
    dir_df = compute_directional_slices(pdf, value_cols=feature_cols, n_slices=n_slices)
    # 将方向特征合并（取示例中已计算的列）
    for c in dir_df.columns:
        if c.startswith(tuple([f for f in feature_cols])):
            X[c] = dir_df[c].values
    X = X.fillna(0)
    return X

def main(args):
    # 读 Parquet（或多个 parquet 分区）
    df = dd.read_parquet(args.input_parquet)
    # 确保分区键按 well_id 以便于 group 操作
    # 简单按 well_id 分区（如果数据量大可事先按 well_id 分区写入）
    # 对 groupby-apply，使用 map_partitions 或 groupby.apply 如果 small groups
    feature_cols = args.features.split(",")
    # 使用 map_partitions 示例（对每个分区内按井排序做滚动）
    def _apply_partition(part):
        return make_features_partition(part, feature_cols, window=args.window, n_slices=args.n_slices)

    feats = df.map_partitions(_apply_partition, meta={c: 'f8' for c in feature_cols})
    # 将原关��列带回（well_id depth target）
    # 注意：这里仅示例，真实场景需严格构造 meta 与列
    out = dd.concat([df[['well_id','depth']].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out.to_parquet(args.output_parquet, write_index=False)
    print("Saved features to", args.output_parquet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=str, required=True)
    parser.add_argument("--output-parquet", type=str, required=True)
    parser.add_argument("--features", type=str, default="GR,RHOB,NPHI,DT,RES")
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--n-slices", type=int, default=8)
    args = parser.parse_args()
    main(args)
