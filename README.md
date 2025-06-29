# 3D Gaussian Splatting as Markov Chain Monte Carlo (Packaged Python Version)

This repository contains the **refactored Python code for [3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc)**. It is forked from commit [7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13](https://github.com/ubc-vision/3dgs-mcmc/tree/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13). The original code has been **refactored to follow the standard Python package structure**, while **maintaining the same algorithms as the original version**.

## Features

* [x] Code organized as a standard Python package
* [x] Markov Chain Monte Carlo trainer for 3D Gaussian Splatting

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

## Install (Development)

Install [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting).
You can download the wheel from [PyPI](https://pypi.org/project/gaussian-splatting/):
```shell
pip install --upgrade gaussian-splatting
```
Alternatively, install the latest version from the source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

(Optional) If you prefer not to install `gaussian-splatting` and `reduced-3dgs` in your environment, you can install them in your `lapis-gs` directory:
```sh
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

# ** NeurIPS 2024 SPOTLIGHT **
# 3D Gaussian Splatting as Markov Chain Monte Carlo

[![button](https://img.shields.io/badge/Project%20Website-orange?style=for-the-badge)](https://ubc-vision.github.io/3dgs-mcmc/)
[![button](https://img.shields.io/badge/Paper-blue?style=for-the-badge)](https://arxiv.org/abs/2404.09591)
[![button](https://img.shields.io/badge/Video-green?style=for-the-badge)](https://neurips.cc/virtual/2024/poster/94984)

<span class="author-block">
  <a href="https://shakibakh.github.io/">Shakiba Kheradmand</a>,
</span>
<span class="author-block">
  <a href="http://drebain.com/"> Daniel Rebain</a>,
</span>
<span class="author-block">
  <a href="https://hippogriff.github.io/"> Gopal Sharma</a>,
</span>
<span class="author-block">
  <a href="https://wsunid.github.io/"> Weiwei Sun</a>,
</span>
<span class="author-block">
  <a href="https://scholar.google.com/citations?user=1iJfq7YAAAAJ&hl=en"> Yang-Che Tseng</a>,
</span>
<span class="author-block">
  <a href="http://www.hossamisack.com/">Hossam Isack</a>,
</span>
<span class="author-block">
  <a href="https://abhishekkar.info/">Abhishek Kar</a>,
</span>
<span class="author-block">
  <a href="https://taiya.github.io/">Andrea Tagliasacchi</a>,
</span>
<span class="author-block">
  <a href="https://www.cs.ubc.ca/~kmyi/">Kwang Moo Yi</a>
</span>

<hr>

<video controls>
  <source src="docs/resources/training_rand_compare/bicycle_both-rand.mp4" type="video/mp4">
</video>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{kheradmand20243d,
    title = {3D Gaussian Splatting as Markov Chain Monte Carlo},
    author = {Kheradmand, Shakiba and Rebain, Daniel and Sharma, Gopal and Sun, Weiwei and Tseng, Yang-Che and Isack, Hossam and Kar, Abhishek and Tagliasacchi, Andrea and Yi, Kwang Moo},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2024},
    note = {Spotlight Presentation},
   }</code></pre>
  </div>
</section>


## Updates
### Dec. 5th, 2024
A new change has been pushed to diff-gaussian-rasterization. In order to pull it:
```sh
cd submodules/diff-gaussian-rasterization
git pull origin gs-mcmc
cd ../..
pip install submodules/diff-gaussian-rasterization
```

This change incorporates "Section B.2 Tighter Bounding of 2D Gaussians" from [StopThePop](https://arxiv.org/abs/2402.00525) paper. This bound allows to fit a tighter bound around Gaussians when opacity is less than 1.

## How to Install

This project is built on top of the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) and has been tested only on Ubuntu 20.04. If you encounter any issues, please refer to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) for installation instructions.

### Installation Steps

1. **Clone the Repository:**
   ```sh
   git clone --recursive https://github.com/ubc-vision/3dgs-mcmc.git
   cd 3dgs-mcmc
   ```
2. **Set Up the Conda Environment:**
    ```sh
    conda create -y -n 3dgs-mcmc-env python=3.8
    conda activate 3dgs-mcmc-env
    ```
3. **Install Dependencies:**
    ```sh
    pip install plyfile tqdm torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    conda install cudatoolkit-dev=11.7 -c conda-forge
    ```
4. **Install Submodules:**
    ```sh
    CUDA_HOME=PATH/TO/CONDA/envs/3dgs-mcmc-env/pkgs/cuda-toolkit/ pip install submodules/diff-gaussian-rasterization submodules/simple-knn/
    ```
### Common Issues:
1. **Access Error During Cloning:**
If you encounter an access error when cloning the repository, ensure you have your SSH key set up correctly. Alternatively, you can clone using HTTPS.
2. **Running diff-gaussian-rasterization Fails:**
You may need to change the compiler options in the setup.py file to run both the original and this code. Update the setup.py with the following extra_compile_args:
    ```sh
    extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
    ```
    Afterwards, you need to reinstall diff-gaussian-rasterization. This is mentioned in [3DGS-issue-#41](https://github.com/graphdeco-inria/gaussian-splatting/issues/41).
    
By following these steps, you should be able to install the project and reproduce the results. If you encounter any issues, refer to the original 3DGS code base for further guidance.

## How to run
Running code is similar to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) with the following differences:
- You need to specify the maximum number of Gaussians that will be used. This is performed using --cap_max argument. The results in the paper uses the final number of Gaussians reached by the original 3DGS run for each shape.
- You need to specify the scale regularizer coefficient. This is performed using --scale_reg argument. For all the experiments in the paper, we use 0.01.
- You need to specify the opacity regularizer coefficient. This is performed using --opacity_reg argument. For Deep Blending dataset, we use 0.001. For all other experiments in the paper, we use 0.01.
- You need to specify the noise learning rate. This is performed using --noise_lr argument. For all the experiments in the paper, we use 5e5.
- You need to specify the initialization type. This is performed using --init_type argument. Options are random (to initialize randomly) or sfm (to initialize using a pointcloud).

## How to Reproduce the Results in the Paper
```sh
python train.py --source_path PATH/TO/Shape --config configs/shape.json --eval
```




