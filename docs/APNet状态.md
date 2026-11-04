# AP-Net 基线的当前状态

本包没有将其他网络改名为 AP-Net，也没有填入 AP-Net 论文中的结果。

已查看作者主页指向的公开仓库：

- [作者主页](https://luyuhuan.github.io/)
- [代码](https://github.com/Luyuhuan/AP-Net)
- 核对时的 commit：`41c52935bc5aa2534fd4c0a5f2a2300f5066d7f9`

仓库入口 `unimatch_cropw.py` 需要额外 `shapepriors-path`、先验尺寸与 K；示例脚本含原作者环境绝对路径，部分裁剪逻辑按原任务类别数写定。它不是可以直接放到当前成人 CAMUS 四类任务和 Windows 单 GPU 环境中运行的已适配版本。

当前已完成并可运行：ACES、监督 U-Net、Mean Teacher、从头训练的 SegFormer-B0、监督图、置信度图、Gaussian 图、共享视图图控制。

**AP-Net 尚未形成经核验的独立复现，因此未进入自动训练列表，结果汇总明确记录此缺项。** 本包的 RUN_ALL 不等于原实验方案所有文献基线均已完成。后续若完成真实适配，应单独记录生成先验的来源、成人标签改动和所有训练配置，再在同一患者划分与标签预算上重训。

没有复制分割论文的性能来补这个空项。也未将作者仓库内容打包为本项目原创代码。
