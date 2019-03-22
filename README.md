# identify parking spots

## Hardware requirements

[NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher.](https://developer.nvidia.com/cuda-gpus)

## Software requirements

- GPU drivers (410.x)
- [CUDA](https://developer.nvidia.com/cuda-90-download-archive) (9.0)
- [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (≥7.4.1 for CUDA 9.0)
- [Anaconda](https://www.anaconda.com/distribution/)

## Command

```bash
# train network
python train-network.py -d ./train_data/train -m v1.model

# test network
python test-network.py -m v1.model -i train_data/test/occupied/pupcr-3-2012-09-20_11_24_28.png
```