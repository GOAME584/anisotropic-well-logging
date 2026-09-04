# Anisotropic Well Logging

这是一个示例仓库，演示用于处理各向异性测井（anisotropic well logging）的特征工程、分布式训练与 Thonny 可视化的流水线。

主要内容（位于 models/anisotropic/）

- MODEL_PAGE.md -- 模型页/说明（包含方法论、特征工程建议、评估与部署指南）
- big_data_pipeline.py -- 基于 Dask 的大数据特征工程示例（生成 Parquet 特征）
- train_dask_xgb.py -- 使用 xgboost.dask 在 Dask 集群或单机并行上训练模型
- thonny_viz.py -- Thonny 友好的可视化脚本：深度曲线、预测 vs 真实、方位玫瑰图
- train.py / infer.py -- 早期的单机示例脚本（CSV 输入）
- requirements.txt -- 推荐依赖列表

快速开始

1) 克隆仓库：

   git clone https://github.com/GOAME584/anisotropic-well-logging.git
   cd anisotropic-well-logging

2) 建议创建虚拟环境并安装依赖：

   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r models/anisotropic/requirements.txt

3) 准备数据

   - 推荐将原始测井数据导出为 Parquet 并按 well_id 分区：
     /data/logs/...
   - 每行至少包含：depth, well_id, GR, RHOB, NPHI, DT, RES, azimuth, inclination, target

4) 生成特征（Dask 单机或集群）

   # 在 Python 中开启 Dask Client（或使用已有集群）
   python -c \"from dask.distributed import Client; Client(n_workers=4)\"

   python models/anisotropic/big_data_pipeline.py \\
     --input-parquet /data/logs/ \\
     --output-parquet /tmp/feats/ \\
     --features GR,RHOB,NPHI,DT,RES \\
     --window 11 --n-slices 8

5) 分布式训练（Dask + XGBoost）

   # 在已有 Dask client 环境下：
   python models/anisotropic/train_dask_xgb.py \\
     --feats /tmp/feats/ \\
     --model-out models/xgb.json \\
     --meta-out models/meta.json \\
     --n-workers 4 --num-boost-round 1000 \\
     --learning-rate 0.05 --max-depth 6

   说明：如使用 dask scheduler 地址，请传 --dask-scheduler tcp://... 参数

6) 单机训练与推理（备用）

   # 单机快速试验（CSV）
   python models/anisotropic/train.py --input_csv sample_well.csv --output_dir models
   python models/anisotropic/infer.py --input_csv sample_well.csv --model_dir models --output_csv preds.csv

7) Thonny 可视化（本地交互）

   # 在 Thonny 中或终端运行：
   python models/anisotropic/thonny_viz.py --input-csv sample_well.csv --model models/xgb.json --meta models/meta.json

建议与注意事项

- 切勿随机打散深度点；CV 应按井或按段分组（GroupKFold）。
- 确保 azimuth/inclination 单位一致（度或弧度）。
- 对大数据场景，优先使用 Parquet 分区并按 well_id 写入以加速 groupby 操作。
- XGBoost 的 Dask 接口需要兼容的 xgboost 版本（>=1.5 建议）。

后续操作（我可以帮助）

- 创建一个 Pull Request 将分支 add/anisotropic-well-logging 合并到 main（我可以替你创建 PR）。
- 为仓库添加 CI（例如 GitHub Actions）做 lint、格式检查或基础单元测试。
- 根据你的数据样本定制脚本参数并跑一次示例训练。

License

本仓库初始化时已附带 MIT 许可证。
