# gauss-splat

## 🔥Latest News

- [2026/10] 新增Ascend 950PR/Ascend 950DT产品支持，新增SIMT渲染、FusedSSIM等算子。
- [2026/07] gauss-splat项目首次上线，开源算子支持Atlas A2/A3系列产品。
- [2026/07] 支持3D Gaussian Splatting核心算子，包括球谐函数、协方差计算、投影、排序、筛选等。

## 🚀概述

gauss-splat是基于[CANN](https://hiascend.com/software/cann)的3D Gaussian Splatting渲染加速库，提供高性能的PyTorch扩展接口，覆盖3DGS训练和推理全流程。本库通过Ascend C算子加速核心计算，在Atlas A2/A3与Ascend 950PR/950DT产品上实现显著性能提升。

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。

## 🛠️环境准备

### 系统配置(推荐)

| 项目 | 本文安装命令使用的组合 |
| :--- | :--- |
| 硬件 | Atlas A2/A3、Ascend 950PR/950DT |
| CANN | **推荐 9.2.0 及以上；以下组合使用 9.2.0** |
| TorchNPU 产品发布版本 | **26.2.0** |
| Python | **3.12** |
| torch | **2.12.0+cpu** |
| torch_npu 安装包 | **2.12.0.post3** |
| torchvision（训练样例） | **0.27.0+cpu** |

`26.2.0` 是昇腾 TorchNPU 的产品发布版本，不是 `torch` 或 `torch_npu` 的 pip 包版本。上述 `torch 2.12.0 / torch_npu 2.12.0.post3 / CANN 9.2.0` 配套关系见[官方版本表](https://www.hiascend.com/developer/download/compatibility)。版本约束集中在 [constraints-cann92.txt](constraints-cann92.txt)；选择其他 CANN/框架组合时，应按官方配套表一起调整，不能只升级其中一个包。

CANN环境准备

> **说明**：仅编译本项目不运行（编译态）时，只需安装CANN toolkit包；运行本项目（运行态）时，需安装NPU驱动、CANN toolkit包与CANN ops包。

1. **安装NPU驱动（运行态依赖）**：使用`npu-smi info`检查是否有NPU相关信息，若没有或无法明确识别芯片型号，请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》完成驱动安装。
2. **安装CANN软件包**：本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），**推荐使用 CANN 9.2.0 及以上版本**，并安装与所选版本及硬件配套的 toolkit 和 ops 包。请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》，选择`稳定版本`，并根据产品系列、CPU架构、操作系统等下载对应软件包（`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-${chip_type}-ops_${version}_linux-${arch}.run`），最后参考页面提供的命令完成安装。

   > **说明**：`npu-smi info`输出可能无法直接体现910C芯片型号。910C所属产品系列为Atlas A3训练推理系列产品，对应的CANN Ops安装包SoC代号为`a3`，即`Ascend-cann-a3-ops_${version}_linux-${arch}.run`。
3. **下载框架 wheel**：从[Ascend Extension for PyTorch 插件页](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)选择 **TorchNPU 产品版 26.2.0** 下适配 **PyTorch 2.12.0、Python 3.12** 和目标 CPU 架构的安装包，准备 `torch 2.12.0+cpu` 与 `torch_npu 2.12.0.post3` 两个 wheel。进入下文的虚拟环境后再安装。

## ⬇️安装

完整流程为“创建环境 → 安装配套框架 → 安装构建依赖 → 编译并安装 gauss-splat → 安装样例依赖 → 运行样例”。将 `/path/to/` 替换为框架 wheel 的实际下载目录，CANN 环境脚本也应使用实际安装路径。

```bash
conda create -n 3dgs python=3.12 -y
conda activate 3dgs

git clone -b ${tag_version} https://gitcode.com/cann/gauss-splat.git
cd gauss-splat
source /usr/local/Ascend/cann/set_env.sh  # 按实际安装路径调整

python -m pip install -c constraints-cann92.txt /path/to/torch-2.12.0+cpu-*.whl /path/to/torch_npu-2.12.0.post3-*.whl
python -m pip install -r requirements-build.txt

# 确认框架版本和 NPU 可用性；有设备的运行环境应输出 True
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.is_available())"

# Atlas A2/A3；A5 将此行替换为 bash build.sh --python=3.12 --a5
bash build.sh --python=3.12
python -m pip install --no-deps --force-reinstall dist/*.whl
```

如果 pip 提示 `Defaulting to user installation`，先用 `python -m pip --version` 确认正在使用已激活且可写的虚拟环境，再继续安装；用户目录与虚拟环境混装可能导致加载另一套 torch/torchvision。

`--python` 应与激活环境一致。也可用 `python -m pip install . --no-build-isolation` 替代最后两步进行默认架构安装。训练/评估额外需要的依赖见下方《测试》或[训练样例](examples/README.md)。

<details>
<summary>编译时找不到 C++ 标准库头文件</summary>

若编译提示 `cstdint`、`cfloat` 或 `type_traits file not found`，请先确认已安装与编译器匹配的 C++ 开发头文件，再启用本项目提供的兼容开关（默认关闭）：

```bash
export ASCENDC_ENABLE_HOST_HEADERS=ON
bash build.sh --python=3.12
```

</details>

> **数据解析依赖**：训练样例使用 `pycolmap>=3.12.6,<5` 和 `numpy>=1.26.0`。若没有适配当前 Python 版本和 CPU 架构的 pycolmap wheel，请参考[官方源码安装说明](https://github.com/colmap/colmap/tree/main/python)。

Atlas A5 请在配套的 A5 CANN 环境中使用 `bash build.sh --python=3.12 --a5` 编译，并参考 [训练样例](examples/README.md)。

安装完成后，请将快速开始代码保存到项目源码目录之外，并从源码目录之外运行。例如，将脚本保存为与`gauss-splat`目录同级的`quick_start.py`后执行：

```bash
cd ..
python quick_start.py
```

请勿在项目根目录下直接运行该脚本。否则Python会优先导入源码目录中的`gauss_splat`包，而不是已安装的包含`gsplat_c`扩展的包，进而导致`ModuleNotFoundError: No module named 'gauss_splat.gsplat_c'`。

## 🚀快速开始

### 完整渲染流程（使用Rasterizer）

示例使用 `rasterization(...)`，相机输入为相机到世界的变换。高斯属性采用连续的 float32 NPU 张量：`means/scales [3,N]`、`quats [4,N]`、`opacities [N]`，3 阶球谐使用单个 `sh [16,3,N]`。

此布局需设置 `splats["is_channel_first"] = True`（如下例），可避免内部布局转换；省略该标记时，接口默认按点优先布局读取：属性为 `[N,D]`，球谐系数为 `[N,K,3]`。

```python
import torch
import torch_npu
import gauss_splat

# 1. 初始化光栅化器
rasterizer = gauss_splat.Rasterizer()

# 2. 准备高斯点云数据
N = 10000  # 高斯点数量
splats = {
    "means": torch.randn(3, N, device='npu:0'),           # 位置
    "quats": torch.randn(4, N, device='npu:0'),           # 四元数
    "scales": torch.randn(3, N, device='npu:0'),          # 缩放（log空间）
    "opacities": torch.randn(N, device='npu:0'),          # 不透明度（logits）
    "sh": torch.randn(16, 3, N, device='npu:0'),          # 0至3阶球谐系数
    "is_channel_first": True,                             # channel-first（推荐，省去内部转换）
}

# 3. 准备相机参数
C = 1  # 相机数量
camtoworlds = torch.eye(4, device='npu:0').unsqueeze(0).expand(C, -1, -1)
Ks = torch.tensor([
    [500.0, 0.0, 960.0],
    [0.0, 500.0, 540.0],
    [0.0, 0.0, 1.0]
], device='npu:0').unsqueeze(0).expand(C, -1, -1)

# 4. 渲染
render_colors, render_depths, info = rasterizer.rasterization(
    cam=(camtoworlds, Ks, "RGB"),
    size=(1920, 1080),
    tile_size=32,
    active_sh_degree=3,
    splats=splats,
    camera_model="pinhole"
)

print(f"渲染颜色形状: {render_colors.shape}")  # (1, 1080, 1920, 3)
print(f"渲染深度形状: {render_depths.shape}")  # (1, 1080, 1920, 1)
```

### 其他

- 完整训练/推理样例可参考 `examples`。
- 更多接口用法参考tests目录下的测试代码。

## 📖Python API

| Python接口名                             | 功能说明                                                                                         |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | 球谐函数计算，支持自动反向传播，将方向向量转换为视角相关颜色                                     |
| `projection_three_dims_gaussian_fused` | 投影筛选融合算子，完整的3D→2D投影流水线（协方差计算+投影变换+高斯筛选），支持自动反向传播       |
| `calc_render`                          | 高斯溅射渲染计算，执行基于Tile的Alpha混合与深度计算，根据编译后端（clip_last_ids / clip_gsids）自动选择实现，支持自动反向传播 |
| `gaussian_sort`                        | 高斯球深度排序，对高斯球按深度进行Tile-based排序，用于渲染流水线                                 |
| `flash_gaussian_build_mask`            | Flash渲染掩码构建，构建Flash Gaussian Splatting渲染所需的掩码和索引                              |
| `gaussian_filter`                      | 高斯球筛选（低级API），独立筛选算子，通常在 `projection_three_dims_gaussian_fused`内部自动调用 |
| `fused_ssim`                           | 融合SSIM（结构相似度）前向/反向计算，用于训练损失，支持自动反向传播                              |
| `Rasterizer`                           | 完整渲染器（推荐），封装完整渲染流水线（投影+排序+Tile构建+渲染计算）                            |
| `get_render_schedule_cpp`              | 渲染调度辅助，获取渲染调度信息（内部辅助函数）                                                   |
| `color_exposure`                       | 高斯球颜色做曝光处理，支持自动反向传播 |

**详细接口说明**： 各API的具体参数和返回值见[API参考文档](docs/zh/gauss_splat_api.md)

## 🔧底层算子接口（ACLNN）

对于需要直接调用底层算子的高级用户，本项目提供ACLNN级别的C++/Python接口。详细API文档见[docs/zh/kernels/](docs/zh/kernels/)。

| 序号 | 算子名称                           | Python接口                                                      | 支持产品   | 文档链接                                                        |
| :--: | :--------------------------------- | :-------------------------------------------------------------- | :--------: | :-------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnSphericalHarmonicsForward.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnSphericalHarmonicsBwd.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnQuatScalesToCovars.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnProjectionThreeDimsGaussianForward.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnGaussianSort.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnGaussianFilter.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFullyFusedProjectionBwd.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      |  A2/A3    | [API文档](docs/zh/kernels/aclnnCalcRenderFwdDoubleClipGsids.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         |  A2/A3    | [API文档](docs/zh/kernels/aclnnCalcRenderBwdVarClipGsids.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFlashGaussianBuildMask.md)             |
|  11  | CalcRenderFwdClipLastids       | `gauss_splat.gsplat_c.calc_render_fwd_clip_last_ids`          |    A5     | [API文档](docs/zh/kernels/aclnnCalcRenderFwdClipLastids.md)           |
|  12  | CalcRenderBwdClipLastids       | `gauss_splat.gsplat_c.calc_render_bwd_clip_last_ids`          |    A5     | [API文档](docs/zh/kernels/aclnnCalcRenderBwdClipLastids.md)           |
|  13  | FusedSsimForward                   | `gauss_splat.gsplat_c.fused_ssim_forward`                     | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFusedSsimForward.md)                   |
|  14  | FusedSsimBackward                  | `gauss_splat.gsplat_c.fused_ssim_backward`                    | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFusedSsimBackward.md)                  |
|  15  | ColorExposureFwd                  | `gauss_splat.gsplat_c.color_exposure_fwd`                    | A5  | [API文档](docs/zh/kernels/aclnnColorExposureFwd.md)                  |
|  16  | ColorExposureBwd                  | `gauss_splat.gsplat_c.color_exposure_bwd`                    | A5  | [API文档](docs/zh/kernels/aclnnColorExposureBwd.md)                  |

**注意**：直接调用底层算子需要手动处理shape变换和内存管理，推荐使用高层Python API。

## 🏗️项目结构

```text
gauss-splat/
├── gauss_splat/            # Python封装层（用户接口）
│   ├── ops/                # 高层Python算子接口
│   │   ├── spherical_harmonics.py       # 球谐函数（自动backward）
│   │   ├── projection_three_dims_gaussian_fused.py  # 投影融合算子
│   │   ├── gaussian_sort.py             # 排序算子
│   │   ├── flash_gaussian_build_mask.py # Flash掩码构建
│   │   ├── gaussian_filter.py           # 筛选算子
│   │   ├── calc_render.py               # 渲染计算（simt / clip_gsids）
│   │   ├── fused_ssim.py                # 融合SSIM（训练损失）
│   │   ├── rendering.py                 # Rasterizer类
│   │   └── get_render_schedule.py       # 辅助函数
│   │   └── color_exposure.py            # 颜色曝光算子
│   ├── csrc/               # C++ binding实现
├── kernels/                # Ascend C算子实现（底层）
│   ├── spherical_harmonics_forward/
│   ├── spherical_harmonics_bwd/
│   ├── quat_scales_to_covars/
│   ├── projection_three_dims_gaussian_forward/
│   ├── gaussian_sort/
│   ├── gaussian_filter/
│   ├── fully_fused_projection_bwd/
│   ├── calc_render_fwd_double_clip_gsids/
│   ├── calc_render_bwd_var_clip_gsids/
│   ├── calc_render_fwd_clip_last_ids/
│   ├── calc_render_bwd_clip_last_ids/
│   ├── fused_ssim_forward/
│   ├── fused_ssim_backward/
│   └── flash_gaussian_build_mask/
│   └── color_exposure_fwd/
│   └── color_exposure_bwd/
├── docs/                   # 文档目录
│   ├── zh/kernels/         # 中文ACLNN算子API文档
│   └── en/kernels/         # 英文ACLNN算子API文档
├── examples/               # 训练/推理样例
├── tests/                  # 测试代码
│   ├── host/               # host(CPU)测试
│   ├── kernels/            # C++算子测试
│   └── ops/                # Python算子测试
└── README.md               # 本文档
```

## 🧪测试

先完成上文的框架和本库安装，并加载当前 CANN 的 `set_env.sh`。运行训练/数据样例时，在仓库根目录执行以下两条安装命令：

```bash
# CPU 源仅用于 torchvision，--no-deps 保留已经安装的 Ascend 框架
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu -c constraints-cann92.txt torchvision
python -m pip install -r examples/requirements.txt

# 检查依赖关系及 torchvision 二进制算子是否能正常加载
python -m pip check
python -c "import torch, torch_npu, torchvision; from torchvision.ops import nms; print(torch.__version__, torch_npu.__version__, torchvision.__version__); print(nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5))"
```

`examples/requirements.txt` 通过 `-r ../requirements-build.txt` 复用构建依赖及其版本约束，只额外声明样例所需的包。公共依赖只维护一次，pip 会复用已满足要求的安装。安装 requirements 不会编译 gauss-splat，仍需先完成上文的编译装包步骤。

constraints 只约束 Python 包，不检查 CANN/驱动版本，也不验证 wheel 来源，因此框架仍需先按上文安装。CPU torchvision 与原来的普通 PyPI 构建不能混用；遇到 `operator torchvision::nms does not exist` 时，可为第一条命令增加 `--force-reinstall` 后重新自检。

`examples/requirements.txt` 已包含数据解析、图像和评估依赖；`examples/gsplat` 随仓库提供，无需再安装同名 pip 包。请在已激活的可写虚拟环境中使用 `python -m pip`，使安装和运行使用同一个解释器。

### Python 测试

`tests/ops/` 为 Python 算子用例（需 NPU），`tests/host/` 为纯 CPU 用例。请从仓库外执行，避免导入源码目录中的 `gauss_splat`：

```bash
REPO=<gauss-splat 仓库绝对路径>; cd /tmp
python -m pytest -v "$REPO/tests/ops"
python -B -m unittest discover -s "$REPO/tests/host" -p 'test_*.py' -v
```

`tests/ops/` 中的用例也可单独运行，例如 `python "$REPO/tests/ops/test_calc_render_fwd.py"`；该目录下的 `perf_benchmark_rasterization.py` 为性能测试脚本，不属于用例。

### C++ 算子测试

```bash
cd <gauss-splat 仓库路径>
bash tests/kernels/build_kernel_tests.sh   # 编译，产物在 tests/kernels/build_out/
bash tests/kernels/run_kernel_tests.sh     # 运行并打印结果
```

> 说明：
> - `build_kernel_tests.sh` 会把算子包内的 `aclnn_*.h` 链接到 `${ASCEND_TOOLKIT_HOME}/include/aclnnop/`。若该目录不可写，链接会失败并报 `aclnnop/...h: No such file or directory`，请改用可写的 include 目录或调整该目录权限。
> - 渲染正/反向算子按架构区分：A2/A3 使用 `test_calc_render_fwd_double_clip_gsids`、`test_calc_render_bwd_var_clip_gsids`；A5 使用 `test_calc_render_clip_fwd_last_ids`、`test_calc_render_bwd_clip_last_ids`。脚本会根据当前构建的 vendor（`customize`/`customize_arch35`）自动跳过不适用的用例。

## 💬更多信息

- [算子API文档](docs/zh/kernels/)：底层ACLNN算子详细说明
- [Python算子测试](tests/ops/)：Python测试代码
- [Host CPU测试](tests/host/)：纯CPU测试代码
- [C++算子测试](tests/kernels/)：C++测试代码（运行方式见上文《测试》）
- [贡献指南](CONTRIBUTING.md)：如何参与项目开发
- [安全声明](SECURITY.md)：安全使用指南
- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/render)：CANN社区render SIG

---

PS：本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues)提交问题。
- **社区互动**：通过GitCode [讨论](https://gitcode.com/cann/gauss-splat/discussions)参与交流。
- **技术专栏**：通过GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki)获取技术文章。
