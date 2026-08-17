#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_glm_include():
    candidates = [
        os.environ.get("GLM_INCLUDE_DIR"),
        os.path.join(sys.prefix, "include"),
        "/usr/local/include",
        "/usr/include",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "glm", "glm.hpp")):
            return candidate
    raise RuntimeError(
        "GLM headers were not found. Install GLM or set GLM_INCLUDE_DIR to the "
        "include directory containing glm/glm.hpp."
    )


glm_include_dir = find_glm_include()

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            include_dirs=[glm_include_dir])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
