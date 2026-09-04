"""
train_dask_xgb.py
使用 Dask + XGBoost 在分布式或单机 Dask 上训练模型。
运行示例（单机）:
  from dask.distributed import Client
  client = Client(n_workers=4)
  python train_dask_xgb.py --feats /tmp/feats/ --model-out models/xgb.json
"""
import argparse
import dask.dataframe as dd
from dask.distributed import Client, wait
import xgboost as xgb
import joblib
import json
import numpy as np

def main(args):
    # 连接 Dask（如果已创建 client 可省略）
    client = Client(args.dask_scheduler) if args.dask_scheduler else Client(n_workers=args.n_workers)
    print("Dask client:", client)

    # 读特征 Parquet（由 big_data_pipeline 输出）
    ddf = dd.read_parquet(args.feats)
    # 假设包含 target 与 well_id
    # 选择特征列（排除 well_id, depth, target）
    exclude = ['well_id','depth','target']
    feature_cols = [c for c in ddf.columns if c not in exclude]

    # 将 Dask DataFrame 转为 DaskDMatrix
    dtrain = xgb.dask.DaskDMatrix(client, ddf[feature_cols], ddf['target'])

    params = {
        'verbosity': 1,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',   # 在分布式上通常选 hist
        'eval_metric': 'rmse',
        'eta': args.learning_rate,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample,
    }
    # 使用 xgboost.dask.train
    output = xgb.dask.train(client, params, dtrain,
                            num_boost_round=args.num_boost_round,
                            evals=[(dtrain, 'train')],
                            early_stopping_rounds=args.early_stopping)
    booster = output['booster']  # Booster object
    booster.save_model(args.model_out)
    # 保存特征列清单
    meta = {'feature_cols': feature_cols}
    with open(args.meta_out, 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved model to", args.model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats", type=str, required=True)
    parser.add_argument("--model-out", type=str, default="models/xgb.json")
    parser.add_argument("--meta-out", type=str, default="models/meta.json")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--dask-scheduler", type=str, default=None)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--early-stopping", type=int, default=50)
    args = parser.parse_args()
    main(args)
