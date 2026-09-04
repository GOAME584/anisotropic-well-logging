# XGBoost 驱动的井测井（Well Logging）模型

版本：0.1.0  
作者：自动生成（可编辑）

## 1. 概述
本模型使用 XGBoost（梯度提升树）对井测井曲线（well logs）进行回归/预测任务（例如：孔隙度预测、声波时差补全、岩性属性回归等）。目标是提供一个可复现、可解释且适合生产部署的基线模型。

## 2. 目标
- 输入：常见井测井曲线（例如：GR、RHOB、NPHI、DT、深度、电阻率等）和深度索引。
- 输出：目标岩石属性（例如孔隙度 porosity、饱和度、声波时差等）。
- 任务类型：回归（可扩展为分类/多任务）。

## 3. 数据说明
推荐数据格式（CSV/Parquet）：
- depth: 浮点或整型（米或英尺）
- GR, RHOB, NPHI, DT, RES, ...: 各类井测井曲线列
- target: 目标值（如 porosity）
示例行：
depth,GR,RHOB,NPHI,DT,RES,target

注意：对于 LAS 文件，先用 lasio 等工具导出到 DataFrame/CSV。

## 4. 预处理建议
- 对齐采样（不同曲线的采样率可能不同，需统一深度网格或插值）。
- 缺失值处理：优先用邻域插值（线性/样条）或基于深度的局部插值；对于大量缺失，可考虑模型预测或删除深度段。
- 深度归一化：保持 depth 列，但不要将深度作为直接尺度因子（可用相对深度/段内归一化）。
- 异常值检测：使用分位数截断或局部统计过滤异常值。
- 单位一致性：确保各测井曲线单位一致（例如 gAPI, g/cc 等）。

## 5. 特征工程（重要）
- 原始曲线值（GR, RHOB, NPHI, DT, ...）
- 局部统计：滑动窗口均值、标准差、中位数（window: 3-11 samples）
- 梯度/导数：一阶、二阶差分（捕捉突变）
- 深度相关特征：depth mod section, relative depth within well/zone
- 窗口能量/方差：反映层内变化
- 哑变量：井号、层段（如果做跨井训练）
- 比值/组合特征：RHOB/NPHI、DT/RHOB 等（物理意义驱动）
- 缩放：对树模型不是必须，但对特征工程统计量做标准化有益（尤其用于后续与线性模型联合时）

## 6. 模型与训练流程
建议使用 XGBoost Regressor：
- API: xgboost.XGBRegressor 或 xgboost.train + DMatrix
- 损失函数：reg:squarederror（或 Huber）; 对异常健壮时可考虑 reg:pseudohubererror
- 验证：基于井/段的分层划分（不要随机打散深度点以避免信息泄露）。常用按井分割训练/验证/测试。
- 交叉验证：GroupKFold（group = well_id）
- 早停：early_stopping_rounds（例如 50）
- 指标：RMSE, MAE, R2；以及按层/按井统计的误差分布

典型超参数：
- n_estimators: 500 - 3000（配合早停）
- learning_rate: 0.01 - 0.1
- max_depth: 3 - 10
- subsample: 0.6 - 1.0
- colsample_bytree: 0.4 - 1.0
- reg_alpha, reg_lambda: 根据过拟合情况调整

选择策略：先做小范围随机搜索或贝叶斯优化，再精调。

## 7. 可解释性与不确定性
- SHAP：用于局部与全局特征重要性解释（TreeExplainer）
- 层/井级别报告：查看模型在不同岩性/层段上的误差分布
- 不确定性估计（选择其一）：
  - 使用模型集合（不同随机种子/子样本）来估计预测方差
  - 使用 Quantile Regression Forest / LightGBM quantile 或者用 XGBoost 的自定义目标估计上下界

## 8. 评价指标
- RMSE, MAE, R2（总体）
- 按井/按层的 RMSE/MAE（查看泛化）
- 分位误差（P10, P50, P90）
- 可视化：预测 vs 真实曲线绘图，误差随深度分布热图，SHAP summary_plot

## 9. 部署与推理
- 将训练好的模型导出（xgboost.save_model -> .json/.bin 或 sklearn wrapper 的 joblib）
- 推理脚本需包含与训练一致的预处理/特征工程步骤
- 小批量推理或流水线：建议使用 FastAPI / Flask 提供 HTTP 接口
- 若在油田实时环境，需考虑内存与延时，可能采用较小的 n_estimators 或更浅的树

## 10. 复现性
- 固定随机种子（numpy, python, xgboost）
- 记录数据版本、特征流水线代码、超参数和训练日志（建议用 MLflow / Weights & Biases）
- 保存模型（JSON/Booster）和预处理器（scaler、特征列定义）

## 11. 注意事项与常见错误
- 切勿随机划分深度点：应按井或按段分组划分，避免信息泄露。
- 不同井的测井仪器/标定差异：考虑对井进行标准化或添加井级别校正因子。
- 数据稀疏/异常：对于稀疏目标（例如少量测井样点有 lab 数据），优先使用基于井的 CV 并扩充特征或使用迁移学习。

## 12. 示例（训练与推理）见配套脚本：train.py, infer.py

## 13. 参考
- XGBoost 官方文档：https://xgboost.readthedocs.io/
- SHAP：https://github.com/slundberg/shap
- LAS 文件处理：lasio
