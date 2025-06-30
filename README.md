# 3D Gaussian Splatting as Markov Chain Monte Carlo (Packaged Python Version)

This repository contains the **refactored Python code for [3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc)**. It is forked from commit [7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13](https://github.com/ubc-vision/3dgs-mcmc/tree/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13). The original code has been **refactored to follow the standard Python package structure**, while **maintaining the same algorithms as the original version**.

## Features

* [x] Code organized as a standard Python package
* [x] Markov Chain Monte Carlo trainer for 3D Gaussian Splatting
* [x] Integration with [reduced-3dgs](https://github.com/yindaheng98/reduced-3dgs)

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

## Install

### PyPI Install

```shell
pip install --upgrade gaussian-splatting-mcmc
```

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

Install [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs).
You can download the wheel from [PyPI](https://pypi.org/project/reduced-3dgs/):
```shell
pip install --upgrade reduced-3dgs
```
Alternatively, install the latest version from the source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
```

(Optional) If you prefer not to install `gaussian-splatting` and `reduced-3dgs` in your environment, you can install them in your `lapis-gs` directory:
```sh
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
```

## Quick Start

1. Download dataset (T&T+DB COLMAP dataset, size 650MB):

```shell
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -P ./data
unzip data/tandt_db.zip -d data/
```

2. Train 3DGS-MCMC:
```shell
python -m gaussian_splatting_mcmc.train -s data/truck -d output/truck -i 30000 --mode base
```

3. Render:
```shell
python -m gaussian_splatting.render -s data/truck -d output/truck -i 30000 --load_camera output/truck/cameras.json
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
