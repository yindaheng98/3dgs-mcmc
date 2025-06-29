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

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

rasterizor_root = "submodules/diff-gaussian-rasterization"
rasterizor_sources = [
    "cuda_rasterizer/utils.cu",
    "rasterize_points.cu",
    "ext.cpp"]

packages = ['mcmc_3dgs'] + ["mcmc_3dgs." + package for package in find_packages(where="mcmc_3dgs")]
rasterizor_packages = {
    'mcmc_3dgs.diff_gaussian_rasterization': 'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization',
}

cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    nvcc_compiler_flags.append("-allow-unsupported-compiler")

setup(
    name="mcmc_3dgs",
    version='0.0.0',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/gaussian-splatting',
    description=u'Refactored python training and inference code for 3D Gaussian Splatting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=packages + list(rasterizor_packages.keys()),
    package_dir={
        'mcmc_3dgs': 'mcmc_3dgs',
        **rasterizor_packages
    },
    ext_modules=[
        CUDAExtension(
            name="mcmc_3dgs.diff_gaussian_rasterization._C",
            sources=[os.path.join(rasterizor_root, source) for source in rasterizor_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags + ["-I" + os.path.join(os.path.abspath(rasterizor_root), "third_party/glm/")]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'gaussian-splatting',
    ]
)
