# Discrete Task-oriented Joint Source-Channel Coding (DT-JSCC)

This is a [Pytorch](https://pytorch.org/docs/stable/index.html) implementation of DT-JSCC for task-oriented communication with digital modulation, as proposed in the paper [Robust Information Bottleneck for Task-Oriented Communication with Digital Modulation](https://arxiv.org/abs/2209.10382).

## Requirements

The codes are compatible with the packages:

* pytorch 1.8.0

* torchvision 0.9.0a0

* numpy 1.23.1

* tensorboardX 2.4

The code can be run on the datasets such as [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), etc. One should download the datasets in a directory (e.g., `./data/`) and change the root parameter in `datasets/dataloader.py`, e.g., 

```python
root = r'./data/
```

## Run experiments

### Training the DT-JSCC model

1. Training the DT-JSCC on the MNIST dataset
   
   `python main.py --dataset MNIST --channels 1 --lam 1e-3 --lr 1e-3 --epoches 400 --latent_d 64 --num_latent 16 --num_embeddings 16 --psnr 4`

2. Training the DT-JSCC on the CIFAR-10 dataset
   
   `python main.py --dataset CIFAR10 --mod psk --lam 1e-3 --lr 1e-3 --epoches 320 --num_embeddings 16 --psnr 4`

The parameter `num_embeddings` is the size of trainable codebook $K$, the parameter `latent_d` is the length of codeword $D$, the `num_latent` is the dimension $d$ of encoded representation $\mathbf{z}$  and the `psnr` is the PSNR of AWGN channel. In the experiments, $Dd = 1024$ for MNIST dataset and $D=512$ and $d=16$ for CIFAR-10 dataset.

### Evaluating the trained DT-JSCC model

`python evaluate.py --dataset CIFAR10 --save_root ./results --name CIFAR10-num_e16-num_latent4-modpsk-snr10.0-lam0.0`  

The parameter `name` is the trained model.

## Citation
```
@article{xie2022robust,
  title={Robust Information Bottleneck for Task-Oriented Communication with Digital Modulation},
  author={Xie, Songjie and Wu, Youlong and Ma, Shuai and Ding, Ming and Shi, Yuanming and Tang, Mingjian},
  journal={arXiv preprint arXiv:2209.10382},
  year={2022}
}
```




