# ACES 实验代码与数据

对应论文：**Effect of anatomical pseudo-label filtering on semi-supervised echocardiographic segmentation: A matched evaluation**。

## 已整理的内容

- 训练、续跑、推理、温度校准、指标计算及 8 种方法的代码。
- 11 个配置、每配置 3 个种子，共 33 次运行的 CSV/JSON 记录。
- 逐帧逐类别、逐患者、逐患者逐类别指标，200 epoch 训练曲线及分层结果。
- 固定的 400/50/50 患者划分与 25% 标注预算的 100 人名单。
- Table 1–4 的复算 CSV、患者配对 bootstrap 和曲线/散点数据。
- Windows 一键入口、依赖、数据字典、软件测试和 GitHub Actions。

原始超声影像、参考标注、预处理图像数组和训练权重未包含在收到的结果包中。本仓库保留数据获取及重建入口；历史结果文件与新训练输出分目录保存。

## 最快核对结果

Windows 双击 **REPRODUCE_RESULTS.bat**。首次会建立分析环境并安装 NumPy；以后使用已安装的环境，无需显卡。

命令行：

    python -m pip install -r requirements-analysis.txt
    python scripts/verify_release.py
    python scripts/reproduce_results.py

输出在 **outputs/reproduced/**。程序重新汇总已有指标、核对患者级聚合、复算配对区间；不会修改 results/archived_runs/ 中的结果。

## 下载数据和重跑训练

1. 从 [CAMUS 官方项目](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)获取影像，放入 data/camus/，保留患者文件夹。你原电脑已经下载好的同一份数据可以复制到这里使用。
2. 按下载版本核对 0=背景、1=左室腔、2=心肌、3=左房，在 settings.json 将 label_map_confirmed 改为 true。
3. data/split_file 已指向 data/splits/camus_patients.json，它是这次实验实际使用的名单。保持此设置可避免重新划分患者。
4. 已装环境可按英文 README 的命令准备数据和训练；Windows 可双击 START.bat，完整队列双击 RUN_ALL.bat。

最初 CAMUS 官方项目描述的是 450 开发/50 测试；本仓库 400/50/50 指收到的实验记录中的实际划分，具体患者清单和原分割文件哈希均已保存。

不要将 data/metadata/manifest_snapshot.json 当作可直接读图的预处理 manifest。它是去除本机路径后的历史快照，实际影像需重新 prepare 生成本机 manifest。

SegFormer-B0 为随机初始化。当前代码固定 FP32；历史 seed17 在 epoch88 从混合精度切到 FP32 的记录已保留。HMC-QU 外验和 AP-Net 核验复现未运行。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| aces/ | 模型、图模块、先验、损失、训练和评估 |
| data/splits/ | 训练/验证/测试及固定标注名单 |
| data/metadata/ | 数据来源、间距、视图、时相、质量及缓存哈希 |
| results/archived_runs/ | 33 次历史运行，保留原数字 |
| results/published/ | 原汇总与配对结果 |
| results/tables/ | 本次由记录复算的表格与绘图 CSV |
| docs/VALIDATION.md | 本次验证范围与结果 |
| docs/UPLOAD_GITHUB_CN.md | 上传步骤 |
| docs/REPRODUCIBILITY.md | 版本、路径处理和复现边界 |

本次整理没有改动模型、损失或训练算法，也没有重跑完整 CAMUS GPU 训练。软件测试与结果复算分别记录，不将小规模软件自检分数当作论文结果。
