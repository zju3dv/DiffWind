# Third-party components

Original DiffWind code is licensed under [Apache 2.0](LICENSE). The components
below and their derived code retain their upstream licenses and copyright notices.

| Component | Included code | License |
| :--- | :--- | :--- |
| [PAC-NeRF](https://github.com/xuan-li/PAC-NeRF) | `simulator/mpm_simulator_force.py` | [MIT, Copyright 2023 Xuan Li](licenses/PAC-NeRF.txt) |
| [Computational Biomimetics of Winged Seeds](https://github.com/leqiqin/Computational-Biomimetics-of-Winged-Seeds) | `HOME_LBM.py` | [MIT, Copyright 2024 leqiqin](licenses/LBM.txt) |
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Gaussian model, cameras, renderer, arguments, and utilities listed below | [Gaussian-Splatting License](third_party/diff-gaussian-rasterization/LICENSE.md), non-commercial research and evaluation |
| [Gaussian-flow rasterizer](https://github.com/Zerg-Overmind/diff-gaussian-rasterization) | `third_party/diff-gaussian-rasterization/` | [Gaussian-Splatting License](third_party/diff-gaussian-rasterization/LICENSE.md), with embedded dependencies listed below |
| PlenOctree spherical harmonics | `utils/sh_utils.py` | BSD 2-Clause, Copyright 2021 The PlenOctree Authors; [full notice in source](utils/sh_utils.py) |
| stb image writer | `third_party/diff-gaussian-rasterization/third_party/stbi_image_write.h` | MIT or public domain; [full notices in source](third_party/diff-gaussian-rasterization/third_party/stbi_image_write.h) |

The 3DGS-derived Python files are `arguments/__init__.py`,
`gaussian_renderer/__init__.py`, `scene/cameras.py`, `scene/gaussian_model.py`,
`utils/camera_utils.py`, `utils/general_utils.py`, `utils/graphics_utils.py`,
and `utils/system_utils.py`. Their original Inria copyright headers are retained.

The rasterizer is based on revision
`2917773c24af4decc298e511c3bf65c5397689fd` of the Gaussian-flow repository.
