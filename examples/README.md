# Atlas A2/A3/A5 3DGS 训练样例

本目录是 Atlas A2/A3/A5 的单卡训练与评估流程。

## 环境与安装

先按[根目录 README](../README.md)完成虚拟环境、Ascend 框架和 gauss-splat 的编译安装。**推荐 CANN 9.2.0 及以上**；本文使用 CANN 9.2.0 / TorchNPU 产品版 26.2.0 / Python 3.12，具体包版本统一记录在 [constraints-cann92.txt](../constraints-cann92.txt)：`torch 2.12.0+cpu`、`torch_npu 2.12.0.post3`、`torchvision 0.27.0+cpu`。

执行顺序为“安装框架和构建依赖 → 编译并安装 gauss-splat → 安装样例依赖 → 运行样例”。完成前两步后，从仓库根目录安装样例依赖：

```bash
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu -c constraints-cann92.txt torchvision
python -m pip install -r examples/requirements.txt
python -m pip check
python -c "import torch, torch_npu, torchvision; from torchvision.ops import nms; print(torch.__version__, torch_npu.__version__, torchvision.__version__); print(nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5))"
cd examples
```

第一条命令只从 CPU 源安装 torchvision。`examples/requirements.txt` 包含 `../requirements-build.txt`，复用构建依赖及版本约束，文件中仅补充样例所需的包；已满足要求的依赖会直接复用。在 `examples/` 目录内也可执行 `python -m pip install -r requirements.txt`，但它不代替 gauss-splat 的编译装包。出现 `operator torchvision::nms does not exist` 时，为第一条安装命令增加 `--force-reinstall` 后重新自检。

默认构建不额外注入主机 C++ 头文件路径；旧环境的可选兼容处理见根目录安装说明。

数据解析使用官方 `pycolmap>=3.12.6,<5` 的 `Reconstruction` API；原 rmbrualla fork 的 `SceneManager` 不再作为依赖。COLMAP 命令行版本与 Python 包版本不同，详见根目录 README。

## 数据与运行

在 `examples` 目录执行：

```bash
python datasets/download_dataset.py
python train.py --data_dir data/360_v2/garden --result_dir results/garden
python train.py --data_dir data/360_v2/garden --result_dir results/garden --ckpt results/garden/ckpts/ckpt_29999_rank0.pt

# 配置 densify 高斯球数量上限
python train.py --data_dir data/360_v2/garden --result_dir results/garden --densify --max-n-gaussian 1000000

# 切换到 channel-first 布局（默认是 N-first）
python train.py --data_dir data/360_v2/garden --result_dir results/garden --is-channel-first
```

### 参数布局开关

训练脚本默认使用 **N-first** 布局（`is_channel_first=False`）：`means/quats/scales` 为 `[N,D]`，`sh` 为 `[N,K,3]`。如需 channel-first（`[D,N]`、`[K,3,N]`），追加 `--is-channel-first`。

如果数据本身已经是 channel-first，建议直接开启 `--is-channel-first`，避免算子内部做布局转换，性能更好。

Checkpoint 始终以 N-first 规范格式保存，加载时按当前开关自动转换；PLY 导出也始终写 N-first。

### `max_n_gaussian` 作用说明

`--max-n-gaussian`（对应 `Config.max_n_gaussian`）用于覆盖 densify Strategy 的 `cap_max`：

- 仅在 `--densify` 开启时生效。
- `max_n_gaussian > 0`：覆盖 `strategy.cap_max`；`max_n_gaussian <= 0`（默认）：沿用 Strategy 默认值（ADC 6,000,000 / Default 3,000,000）。

### 高斯球 DFX 日志说明

```text
[GS]init rank=<rank> world_size=<world_size> num_GS=<初始化点数> strategy=<adc|default|off> max_n_gaussian=<实际生效上限|off> (cfg.max_n_gaussian=<CLI原始值>)
[DENSIFY][config] rank=<rank> world_size=<world_size> strategy=<adc|default> refine=[<start>-<stop>]/<every> reset_every=<reset_every> pause_refine_after_reset=<pause> grow_grad2d=<grow_grad2d> prune_opa=<prune_opa>
[DENSIFY] rank=<rank> step=<step> strategy=<adc|default> before=<density前点数> removed=<prune+capped> added=<dup+split> net=<after-before> after=<density后点数> cap(dup=<clone数> split=<split数> prune=<常规裁剪数> capped=<硬上限裁剪数>)
[GS]step=<step> rank=<rank> num_GS=<当前点数> max_n_gaussian=<实际生效上限|off>
```

- `[GS]init`：每个 rank 训练启动时打印一次，包含 `rank/world_size`；`max_n_gaussian` 为 CLI 覆盖后的实际值，括号内保留 `cfg.max_n_gaussian` 原值；关闭 densify 时策略为 `off`。
- `[DENSIFY][config]`：开启 densify 时每个 rank 打印一次，记录实际生效的 refine 窗口、周期、reset 周期和增密阈值，便于与 `adjust_steps` 后的配置核对。
- `[GS]step`：每个 rank 按 `tb_every` 周期打印，`num_GS` 为该 rank 的本地数量，用于多卡训练观察各 rank 的高斯球数量变化。
- `[DENSIFY]`：每个 rank 在每个 densify refine 步骤打印一次，`added=dup+split`（split 为净增 1），`removed=prune+capped`，`net=after-before`。
- `cap(...)`：本步 clone 复制数、split 数、常规裁剪数（不透明度/屏幕半径/尺度）和被硬上限裁掉的点数。

`train_all.sh` 用于按脚本中的场景列表批量训练。`rasterization/config.py` 定义训练配置，`rasterization/runner.py` 驱动训练与评估。

## 引用

```bibtex
@article{kerbl3Dgaussians,
    author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal={ACM Transactions on Graphics},
    volume={42},
    number={4},
    year={2023}
}

@article{ye2025gsplat,
    title={gsplat: An open-source library for Gaussian splatting},
    author={Ye, Vickie and Li, Ruilong and Kerr, Justin and others},
    journal={Journal of Machine Learning Research},
    volume={26},
    year={2025}
}
```
