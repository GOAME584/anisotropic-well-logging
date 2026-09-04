# anisotropic-well-logging
Anisotropic well logging models, Dask + XGBoost pipelines and Thonny visualizations
概述
本模型专注于利用测井曲线结合井斜/方位等几何信息来建模各向异性响应（例如在有明显地层倾角或井眼偏斜情况下的物性预测）。针对大规模数据，提供基于 Dask 的分布式特征工程和基于 XGBoost 的分布式训练示例；并提供 Thonny 友好的可视化脚本用于快速验算与展示。

目标
输入：常见测井曲线（GR, RHOB, NPHI, DT, RES 等）+ geometry（depth, inclination, azimuth, tool_offset 等）
输出：目标岩石属性（孔隙度、声波时差、岩性概率等）
任务类型：回归/分类/多任务
各向异性要点（特征思想）
方向性特征：
inclination（井斜）、azimuth（方位）
沿井周向/方位分段统计：按方位切片的平均测井响应（例如将周围环向分为 N 段，计算每段均值/方差）
方向导数/张量：基于方位对测井响应做旋转不变或旋转敏感描述
井眼偏差与层向角度：相对地层倾角差值（需要地层或井轨迹信息）
深度与局部统计：
窗口均值/方差、梯度（一阶/二阶）、频谱能量（短时傅里叶/小波）
物理组合特征：RHOB/NPHI, DT/RHOB, 电阻率比值等
井级与测线级标识：well_id、survey_id、tool_type（用于分组 CV）
数据准备与大数据策略
数据格式：Parquet 推荐（列式、支持分区）；CSV 可用但在大数据场景慢。
必要列：depth, well_id, GR, RHOB, NPHI, DT, RES, inclination, azimuth, target
缺失处理：Dask map_partitions + fill/interpolate；对分布式滚动窗口使用 map_overlap。
分区策略：按 well_id 或按地理区域分区，便于 GroupKFold。
大数据处理工具：
Dask (推荐) — 兼容 pandas API, 支持 xgboost.dask
PySpark — 在已有 Hadoop / Spark 集群时可用
采样策略：先用井级采样做快速实验；再用全量数据做最终训练或增量微调。
模型与训练
基线：XGBoost（xgboost.dask 用于分布式）
损失：reg:squarederror / reg:pseudohubererror；分类用 binary:logistic / multi:softprob
CV：GroupKFold（group = well_id 或 survey_id），并同时考虑 azimuth 分层（可用 group + bucketed azimuth）
早停：early_stopping_rounds，监控按井汇总的 RMSE
不确定性：使用模型集合（多次训练）或基于 quantile 模型估计上下界
可解释性
SHAP（TreeExplainer）
层/方位分段误差分布（按方位切片看模型性能）
方向性敏感性测试：通过改变 azimuth/inclination 输入来测试预测响应变化
部署与推理
保存模型：XGBoost JSON/BIN 或 sklearn wrapper 的 joblib
推理管线：与训练一致的分布式特征工程（Dask），采用批处理或流式（按井/区域）
Thonny：用于轻量可视化或本地验证；生产部署用 FastAPI / batch workers
评价指标
RMSE, MAE, R2（总体）
按井/按层/按方位段 RMSE（查看各向异性效果）
P10/P50/P90 / calibration plots
可视化（Thonny 友好）
深度曲线堆叠图（多曲线在深度轴）
预测 vs 真实曲线
方位玫瑰图（anisotropy rose）：展示不同方位段测井/误差分布
SHAP summary / dependence（保存为 png）
运行示例
用 Dask 特征工程： python big_data_pipeline.py --input-parquet /data/logs/ --output-parquet /tmp/feats/
分布式训练： python train_dask_xgb.py --feats /tmp/feats/ --model-out models/xgb.json
在 Thonny 中可视化： python thonny_viz.py --input-csv some_well.csv --model models/xgb.json
注意事项
切勿随机打散深度点，必须按井或按段划分 CV。
方位/井斜角度要统一单位（度或弧度）。
如果测井工具在井眼不同方位有偏差，务必记录 tool_orientation 或做井内校正。
Dask + XGBoost 文档：https://xgboost.readthedocs.io/en/stable/dask.html
SHAP：https://github.com/slundberg/shap
LAS 与测井处理：lasio
