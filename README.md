### Introduction
* This repository contains the PyTorch implementation of deformabel covolution network, including v1 and v2 version. The origin papers can be found from ![here (v1)](https://arxiv.org/abs/1703.06211) and ![here (v2)](https://arxiv.org/abs/1811.11168). And the official implementation is ![here](https://github.com/msracver/Deformable-ConvNets).
* Only deformable convolution is implemented. The deformable RoIPooling is not yet implemented.
* This repository supports PyTorch < 0.4.1, and the codes have been tested with PyTorch = 0.3.0. And you can get PyTorch > 0.4.0 version from ![here](https://github.com/liuqk3/Deformable-ConvNets-PyTorch/tree/master).
* Any questions are welcomed.
### Installation and usage
* The installation of `Deformable-ConvNets` is much easy:
```
$ git clone https://github.com/liuqk3/Deformable-ConvNets-PyTorch.git
$ unzip Deformable-ConvNets-PyTorch.zip
$ cd Deformable-ConvNets-PyTorch
$ sh install.sh
```
* To use the `Deformable-ConvNets`, see the example in `./test.py`.
### To-do list
* Implement deformable RoIPooling.
* ~~Support PyTorch > 0.4, including PyTorch 1.0.~~
