# Analysis of Parking Spaces

## Structure

```bash
.
├── network
|   ├── lenet.py
|   └── tinyvgg.py
├── src_img
|   ├── PUCPR
|   ├── UFPR04
|   └── UFPR05
├── train_data
|   ├── models
|   ├── test_img
|   ├── test_seg
|   └── train
├── calc_accuracy.py
├── data_generator.py
├── detect_contour.py
├── test_image.py
├── test_segment.py
├── train_network.py
└── README.md
```

## Hardware requirements

[NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher.](https://developer.nvidia.com/cuda-gpus)

## Software requirements

- GPU drivers (410.x)
- [CUDA](https://developer.nvidia.com/cuda-90-download-archive) (9.0)
- [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (≥7.4.1 for CUDA 9.0)
- [Anaconda](https://www.anaconda.com/distribution/)
- [Parking Lot Database](http://web.inf.ufpr.br/vri/databases/parking-lot-database/)

## Command

```bash
# train network
python train_network.py -d train_data/train/ -m train_data/models/tinyvgg-200.model

python train_network.py -d train_data/train/pucpr -m train_data/models/tinyvgg-pucpr-200.model

# test network
python test_segment.py -m train_data/v1.model -d train_data/test/
```

## Accuracy

>All accuracy tests are based on 200 images

| Parking lot |  Network |Images | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| -- | -- | -- | -- | -- | -- |
| PUCPR | tinyVGG | 100 | 89.58% | 52.59% | 88.90% |
| PUCPR | tinyVGG | 200 | 99.92% | 98.71% | 89.90% |
| PUCPR | tinyVGG | 1000 | 99.99% | 98.10% | 95.94% |
| PUCPR | LeNet | 200 | 99.84% | 97.14% | 84.74% |
| ALL | tinyVGG | 200 * 3 | 99.74% | 98.55% | 99.61% |

## References

[Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)