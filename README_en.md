# gauss-splat

## Latest News

- [2026/10] Added support for Ascend 950PR/Ascend 950DT products, along with SIMT rendering, FusedSSIM, and other operators.
- [2026/07] The gauss-splat project is initially launched, with open-source operators supporting Atlas A2/A3 series products.
- [2026/07] Supports core 3D Gaussian Splatting operators, including spherical harmonics, covariance computation, projection, sorting, and filtering.

## Overview

gauss-splat is a 3D Gaussian Splatting rendering acceleration library based on [CANN](https://hiascend.com/software/cann). It provides high-performance PyTorch extension interfaces that cover the entire 3DGS training and inference workflow. This library accelerates core computations through Ascend C operators, achieving significant performance improvements on Atlas A2/A3 and Ascend 950PR/950DT products.

## Version Compatibility

The source code of this project is released along with CANN software versions. For the mapping between CANN software versions and project tags, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).

## Environment Preparation

### System Configuration (Recommended)

| Component | Combination used by the commands below |
| :--- | :--- |
| Hardware | Atlas A2/A3, Ascend 950PR/950DT |
| CANN | **9.2.0 or later recommended; this combination uses 9.2.0** |
| TorchNPU product release | **26.2.0** |
| Python | **3.12** |
| torch | **2.12.0+cpu** |
| torch_npu package | **2.12.0.post3** |
| torchvision (training examples) | **0.27.0+cpu** |

`26.2.0` is the Ascend TorchNPU product release, not the pip version of `torch` or `torch_npu`. The `torch 2.12.0 / torch_npu 2.12.0.post3 / CANN 9.2.0` mapping is listed in the [official compatibility table](https://www.hiascend.com/developer/download/compatibility). Package constraints are centralized in [constraints-cann92.txt](constraints-cann92.txt). For another CANN/framework combination, update the matching versions together using the official table.

CANN Environment Preparation

> **Note**: When only compiling this project without running it (compile-time state), you only need to install the CANN toolkit package. When running this project (run-time state), you need to install the NPU driver, CANN toolkit package, and CANN ops package.

1. **Install the NPU driver (run-time dependency)**: Run `npu-smi info` to check whether NPU information is displayed. If there is no NPU information or the chip model cannot be clearly identified, refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the driver.
2. **Install CANN software packages**: The execution of this sample depends on the CANN toolkit (cann-toolkit) and CANN binary operator package (cann-kernels). **CANN 9.2.0 or later is recommended**, with toolkit and ops packages matching the selected release and hardware. Refer to [CANN Quick Installation](https://www.hiascend.com/cann/download), select the **stable version**, and download the corresponding packages based on the product series, CPU architecture, and OS (`Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-${chip_type}-ops_${version}_linux-${arch}.run`). Then follow the commands provided on the page to complete the installation.

   > **Note**: The `npu-smi info` output may not directly indicate the 910C chip model. The 910C chip belongs to the Atlas A3 training/inference series products, and its corresponding SoC code in the CANN ops package is `a3`, that is, `Ascend-cann-a3-ops_${version}_linux-${arch}.run`.
3. **Download the framework wheels**: On the [Ascend Extension for PyTorch download page](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download), select **TorchNPU product release 26.2.0**, **PyTorch 2.12.0**, **Python 3.12**, and the target CPU architecture. Prepare the `torch 2.12.0+cpu` and `torch_npu 2.12.0.post3` wheels, then install them inside the environment created below.

## Installation

Follow this order: create the environment, install the matching framework and build dependencies, compile and install gauss-splat, install example dependencies, then run the examples. Replace `/path/to/` with the framework wheel download directory and adjust the CANN environment script to your installation path.

```bash
conda create -n 3dgs python=3.12 -y
conda activate 3dgs

git clone -b ${tag_version} https://gitcode.com/cann/gauss-splat.git
cd gauss-splat
source /usr/local/Ascend/cann/set_env.sh  # Adjust to your installation path

python -m pip install -c constraints-cann92.txt /path/to/torch-2.12.0+cpu-*.whl /path/to/torch_npu-2.12.0.post3-*.whl
python -m pip install -r requirements-build.txt

# Check framework versions and NPU availability (True on a runtime host with a device)
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.is_available())"

# Atlas A2/A3; for A5 replace this line with: bash build.sh --python=3.12 --a5
bash build.sh --python=3.12
python -m pip install --no-deps --force-reinstall dist/*.whl
```

If pip reports `Defaulting to user installation`, verify `python -m pip --version` points into the activated writable environment. Fix the environment before continuing; mixing user-site and environment packages can load a different torch/torchvision build.

`--python` must match the active environment. Alternatively, replace the last two commands with `python -m pip install . --no-build-isolation` to install for the default architecture. Additional training/evaluation dependencies are listed in Testing below and the [training example](examples/README.md).

<details>
<summary>Missing C++ standard library headers during compilation</summary>

If compilation reports `cstdint`, `cfloat`, or `type_traits file not found`, first install the C++ development headers matching your compiler, then enable this project's compatibility option (disabled by default):

```bash
export ASCENDC_ENABLE_HOST_HEADERS=ON
bash build.sh --python=3.12
```

</details>

> **Data parsing dependencies**: Training examples require `pycolmap>=3.12.6,<5` and `numpy>=1.26.0`. If no pycolmap wheel matches your Python version and CPU architecture, follow the [source installation instructions](https://github.com/colmap/colmap/tree/main/python).

For Atlas A5, build in a matching A5 CANN environment with `bash build.sh --python=3.12 --a5`. See the [training example](examples/README.md).

After installation, save the Quick Start code outside the project source directory and run it from outside that directory. For example, save the script as `quick_start.py` next to the `gauss-splat` directory, then run:

```bash
cd ..
python quick_start.py
```

Do not run the script directly from the project root. Otherwise, Python imports the source-tree `gauss_splat` package before the installed package that contains the `gsplat_c` extension, resulting in `ModuleNotFoundError: No module named 'gauss_splat.gsplat_c'`.

## Quick Start

### Complete Rendering Pipeline (Using Rasterizer)

The example uses `rasterization(...)` with camera-to-world matrices. Gaussian attributes are contiguous float32 NPU tensors: `means/scales [3,N]`, `quats [4,N]`, `opacities [N]`, and a single `sh [16,3,N]` tensor for degree 3.

This layout requires `splats["is_channel_first"] = True`, as shown below, and avoids internal layout conversion. If omitted, the interface defaults to N-first inputs: attributes `[N,D]` and SH coefficients `[N,K,3]`.

```python
import torch
import torch_npu
import gauss_splat

# 1. Initialize the rasterizer
rasterizer = gauss_splat.Rasterizer()

# 2. Prepare Gaussian point cloud data
N = 10000  # Number of Gaussian points
splats = {
    "means": torch.randn(3, N, device='npu:0'),           # Positions
    "quats": torch.randn(4, N, device='npu:0'),           # Quaternions
    "scales": torch.randn(3, N, device='npu:0'),          # Scales (log space)
    "opacities": torch.randn(N, device='npu:0'),          # Opacities (logits)
    "sh": torch.randn(16, 3, N, device='npu:0'),          # SH coefficients for degrees 0 to 3
    "is_channel_first": True,                             # channel-first (recommended, avoids conversion)
}

# 3. Prepare camera parameters
C = 1  # Number of cameras
camtoworlds = torch.eye(4, device='npu:0').unsqueeze(0).expand(C, -1, -1)
Ks = torch.tensor([
    [500.0, 0.0, 960.0],
    [0.0, 500.0, 540.0],
    [0.0, 0.0, 1.0]
], device='npu:0').unsqueeze(0).expand(C, -1, -1)

# 4. Render
render_colors, render_depths, info = rasterizer.rasterization(
    cam=(camtoworlds, Ks, "RGB"),
    size=(1920, 1080),
    tile_size=32,
    active_sh_degree=3,
    splats=splats,
    camera_model="pinhole"
)

print(f"Rendered color shape: {render_colors.shape}")  # (1, 1080, 1920, 3)
print(f"Rendered depth shape: {render_depths.shape}")  # (1, 1080, 1920, 1)
```

### Others

- Complete training/inference samples are available in `examples`.
- For more interface usage, refer to the test code in the tests directory.

## Python API

| Python Interface Name                   | Description                                                                                        |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | Spherical harmonics computation, supports automatic backward propagation, converts direction vectors to view-dependent colors                                     |
| `projection_three_dims_gaussian_fused` | Projection filtering fused operator, complete 3D-to-2D projection pipeline (covariance computation + projection transformation + Gaussian filtering), supports automatic backward propagation       |
| `calc_render`                          | Gaussian splatting rendering computation, performs tile-based alpha blending and depth computation, automatically selecting the implementation based on the built backend (clip_last_ids / clip_gsids), supports automatic backward propagation |
| `gaussian_sort`                        | Gaussian sphere depth sorting, performs tile-based depth sorting on Gaussian spheres for the rendering pipeline                                 |
| `flash_gaussian_build_mask`            | Flash rendering mask construction, builds the masks and indices required for Flash Gaussian Splatting rendering                              |
| `gaussian_filter`                      | Gaussian sphere filtering (low-level API), independent filtering operator, typically called automatically within `projection_three_dims_gaussian_fused` |
| `fused_ssim`                           | Fused SSIM (structural similarity) forward/backward computation for training loss, supports automatic backward propagation                              |
| `Rasterizer`                           | Complete rasterizer (recommended), encapsulates the complete rendering pipeline (projection + sorting + tile construction + rendering computation)                            |
| `get_render_schedule_cpp`              | Rendering schedule helper, obtains rendering schedule information (internal helper function)                                                   |

**Detailed Interface Description**: For specific parameters and return values of each API, refer to the [API Reference Documentation](docs/en/gauss_splat_api_en.md).

## Underlying Operator Interface (ACLNN)

For advanced users who need to directly call underlying operators, this project provides ACLNN-level C++/Python interfaces. For detailed API documentation, refer to [docs/en/kernels/](docs/en/kernels/).

| No. | Operator Name                      | Python Interface                                                | Supported Products | Documentation Link                                               |
| :--: | :--------------------------------- | :-------------------------------------------------------------- | :----------------: | :-------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnSphericalHarmonicsForward_en.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnSphericalHarmonicsBwd_en.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnQuatScalesToCovars_en.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnProjectionThreeDimsGaussianForward_en.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnGaussianSort_en.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnGaussianFilter_en.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFullyFusedProjectionBwd_en.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      | A2/A3             | [API Documentation](docs/en/kernels/aclnnCalcRenderFwdDoubleClipGsids_en.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         | A2/A3             | [API Documentation](docs/en/kernels/aclnnCalcRenderBwdVarClipGsids_en.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFlashGaussianBuildMask_en.md)             |
|  11  | CalcRenderFwdClipLastids       | `gauss_splat.gsplat_c.calc_render_fwd_clip_last_ids`          | A5                | [API Documentation](docs/en/kernels/aclnnCalcRenderFwdClipLastids_en.md)                |
|  12  | CalcRenderBwdClipLastids       | `gauss_splat.gsplat_c.calc_render_bwd_clip_last_ids`          | A5                | [API Documentation](docs/en/kernels/aclnnCalcRenderBwdClipLastids_en.md)                |
|  13  | FusedSsimForward                   | `gauss_splat.gsplat_c.fused_ssim_forward`                     | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFusedSsimForward_en.md)                   |
|  14  | FusedSsimBackward                  | `gauss_splat.gsplat_c.fused_ssim_backward`                    | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFusedSsimBackward_en.md)                  |

**Note**: Directly calling underlying operators requires manual handling of shape transformations and memory management. Using the high-level Python API is recommended.

## Project Structure

```text
gauss-splat/
├── gauss_splat/            # Python wrapper layer (user interface)
│   ├── ops/                # High-level Python operator interfaces
│   │   ├── spherical_harmonics.py       # Spherical harmonics (automatic backward)
│   │   ├── projection_three_dims_gaussian_fused.py  # Projection fused operator
│   │   ├── gaussian_sort.py             # Sorting operator
│   │   ├── flash_gaussian_build_mask.py # Flash mask construction
│   │   ├── gaussian_filter.py           # Filtering operator
│   │   ├── calc_render.py               # Rendering computation (simt / clip_gsids)
│   │   ├── fused_ssim.py                # Fused SSIM (training loss)
│   │   ├── rendering.py                 # Rasterizer class
│   │   └── get_render_schedule.py       # Helper functions
│   ├── csrc/               # C++ binding implementation
├── kernels/                # Ascend C operator implementation (low-level)
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
├── docs/                   # Documentation directory
│   ├── zh/kernels/         # Chinese ACLNN operator API documentation
│   └── en/kernels/         # English ACLNN operator API documentation
├── examples/               # Training/inference samples
├── tests/                  # Test code
│   ├── host/               # Host (CPU) tests
│   ├── kernels/            # C++ operator tests
│   └── ops/                # Python operator tests
└── README.md               # This document
```

## Testing

Install the framework and library as described above, and load `set_env.sh` for your CANN installation. For training/data examples, run these two installation commands from the repository root:

```bash
# Use the CPU index only for torchvision; --no-deps preserves the installed Ascend framework
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu -c constraints-cann92.txt torchvision
python -m pip install -r examples/requirements.txt

# Check dependency metadata and the torchvision binary operator
python -m pip check
python -c "import torch, torch_npu, torchvision; from torchvision.ops import nms; print(torch.__version__, torch_npu.__version__, torchvision.__version__); print(nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5))"
```

`examples/requirements.txt` includes `../requirements-build.txt` via `-r`, reusing its build dependencies and version constraints, and lists only additional example packages. Common dependencies are maintained once; pip reuses installed packages that satisfy the requirements. Installing requirements does not compile gauss-splat, so complete the build and package installation above first.

Constraints cover Python packages, not CANN/driver versions or wheel provenance, so install the framework as instructed above first. The CPU torchvision build is not interchangeable with the original plain PyPI build. For `operator torchvision::nms does not exist`, add `--force-reinstall` to the first command and rerun the checks.

`examples/requirements.txt` includes data, image, and evaluation dependencies. `examples/gsplat` is vendored in the repository; no separate package with that name is needed. Use `python -m pip` inside an activated, writable environment so installation and execution use the same interpreter.

### Python Tests

`tests/ops/` contains the Python operator cases (requires NPU) and `tests/host/` contains pure CPU cases. Run them outside the repository to avoid importing the source-tree `gauss_splat` package:

```bash
REPO=<absolute path to the gauss-splat repository>; cd /tmp
python -m pytest -v "$REPO/tests/ops"
python -B -m unittest discover -s "$REPO/tests/host" -p 'test_*.py' -v
```

The cases in `tests/ops/` can also be run individually, for example `python "$REPO/tests/ops/test_calc_render_fwd.py"`; `perf_benchmark_rasterization.py` in that directory is a performance benchmark script, not a test case.

### C++ Operator Tests

```bash
cd <gauss-splat repository path>
bash tests/kernels/build_kernel_tests.sh   # Build; binaries go to tests/kernels/build_out/
bash tests/kernels/run_kernel_tests.sh     # Run and print results
```

> Notes:
> - `build_kernel_tests.sh` links the package's `aclnn_*.h` into `${ASCEND_TOOLKIT_HOME}/include/aclnnop/`. If that directory is not writable the link fails with `aclnnop/...h: No such file or directory`; use a writable include directory or adjust the directory permissions.
> - The render forward/backward operators are architecture-specific: A2/A3 uses `test_calc_render_fwd_double_clip_gsids` and `test_calc_render_bwd_var_clip_gsids`, while A5 uses `test_calc_render_fwd_clip_last_ids` and `test_calc_render_bwd_clip_last_ids`. The script automatically skips the cases that do not apply to the current build vendor (`customize`/`customize_arch35`).

## More Information

- [Operator API Documentation](docs/en/kernels/): Detailed description of underlying ACLNN operators
- [Python Operator Tests](tests/ops/): Python test code
- [Host CPU Tests](tests/host/): Pure CPU test code
- [C++ Operator Tests](tests/kernels/): C++ test code (see the Testing section above for how to run)
- [Contribution Guide](CONTRIBUTING.md): How to participate in project development
- [Security Statement](SECURITY.md): Security usage guide
- [License](LICENSE): CANN Open Software License Agreement Version 2.0
- [SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/render): CANN community render SIG

---

Note: The features and documentation of this project are being continuously updated and improved. Stay tuned for the latest version.

- **Issue Feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues).
- **Community Interaction**: Participate in discussions through GitCode [Discussions](https://gitcode.com/cann/gauss-splat/discussions).
- **Technical Articles**: Access technical articles through GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki).
