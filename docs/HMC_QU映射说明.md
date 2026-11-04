# HMC-QU 外部映射

提供方数据可能有不同版本。代码不根据相似文件名或视频帧数猜患者、掩膜变量名、裁剪区域与帧偏移。必须在运行前核对，不能根据结果高低选择映射。

`data/hmc_qu/external_manifest.template.csv` 只是模板，不会自动运行。填完真实记录后另存为 `external_manifest.csv`，每个已核对的带标注帧一行。

| 字段 | 含义 |
|---|---|
| patient_id | 真实匿名患者标识；同患者所有记录相同 |
| recording_id | 视频/记录标识；患者内先按记录、再按患者等权汇总 |
| image / mask | 相对 CSV 所在文件夹的路径，或绝对路径 |
| image_frame / mask_frame | 多帧文件中所需帧，**从 0 起**；单幅 2D 文件留空 |
| image_key / mask_key | `.mat` 或 `.npz` 内变量名；MAT 必须填写，不猜第一个变量 |
| image_frame_axis / mask_frame_axis | `.mat/.npy/.npz` 的帧维，默认 0；例如 H×W×T 的掩膜填 2 |
| foreground_value | 真实前景像素值，例如 1 或 255；其余必须为 0 |
| image_grid_transform | 默认 `none`。仅在提供方说明图像与标注是同一完整视野的直接缩放时，可填 `resize_to_mask` |
| alignment_basis | 使用上述缩放时必须填写来源说明；如果掩膜对应裁剪 ROI，需要先按真实 ROI 转换，不能直接全图缩放 |
| spacing_x / spacing_y | 可靠原生物理间距，未知留空；本实现外部表面距离仍统一报 pixel |
| phase | 已知 ED/ES 可填；未知或全周期可留空，代码标为 CYCLE，不据预测推断 |
| confirmed | 全部核对完填 `true`；模板默认为 false，避免直接跑错 |

支持 `.nii/.nii.gz/.mhd/.mha`、`.npy/.npz`、普通灰度图片、通过 scipy 读取的 MAT，以及 OpenCV 支持的视频。MAT v7.3/HDF5、复杂对象结构或特殊编码需先按其真实规范转换；不宣称能自动解码所有医学格式。

二类目标：

- `myocardium`：参考前景为心肌组织，只取 CAMUS 心肌概率。
- `epicardial_area`：参考前景是心外膜包围的实心区域，取左室腔与心肌概率之和。

前景概率≥0.5 作为统一二类预测规则；不是把 CAMUS 三类 argmax 随意解释为外部三类标签。两种目标不能混在同一表里。无法确认时先不运行外部验证。

HMC-QU 的图像、标签、患者 ID、先验、阈值均不参与 CAMUS 训练。此处核对的是标注语义与坐标映射，不进行性能驱动的模型选择。
