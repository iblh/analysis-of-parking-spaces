# Analysis of Parking Spaces

## Structure

```bash
.
├── network/
|   ├── lenet.py
|   └── minivgg.py
├── src_img/
|   ├── PUCPR/
|   ├── UFPR04/
|   └── UFPR05/
├── train_data/
|   ├── models/
|   ├── test_img/
|   ├── test_seg/
|   └── train/
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

## Datasets

- [Parking Lot Database](http://web.inf.ufpr.br/vri/databases/parking-lot-database/)
  - PUCPR: 4474 pics
  - UFPR04: 3791 pics
  - UFPR05: 4152 pics

## Command

```bash
# Train network (all datasets)
python train_network.py -i train_data/train/ -o train_data/models/minivgg/all

# Train network (specific dataset)
python train_network.py -i train_data/train/pucpr -o train_data/models/minivgg/pucpr

# Test network
python test_segment.py -m train_data/v1.model -d train_data/test/
```

## Accuracy

> All accuracy tests are based on 20,000 parking spaces

| Parking lot | Network | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| ----------- | ------- | ----------- | ------------ | ------------ |
| PUCPR       | AlexNet | 99.87%      | 98.06%       | 94.69%       |
| PUCPR       | LeNet   | 99.84%      | 97.56%       | 84.78%       |
| PUCPR       | MiniVGG | 99.95%      | 99.02%       | 95.77%       |
| PUCPR       | VGG13   | 99.97%      | 98.37%       | 95.13%       |
| PUCPR       | VGG16   | 99.77%      | 96.79%       | 92.13%       |
| ALL         | AlexNet | 99.44%      | 93.01%       | 98.52%       |
| ALL         | LeNet   | 99.58%      | 97.27%       | 98.83%       |
| ALL         | MiniVGG | 99.75%      | 98.80%       | 99.57%       |
| ALL         | VGG13   | 99.67%      | 98.79%       | 99.07%       |
| ALL         | VGG16   | 93.20%      | 77.69%       | 98.87%       |

> PUCPR 36 days
>
> UFPR04 31 days
>
> UFPR05 33 days
>
> About 12 hours (6am - 6pm)
>
> 12 sheets per hour (Take one shot every 5 minutes)
>
> 12x12x0.2=28.8, 33x0.2=6.6, 28.8x6.6=190.08

| Parking lot | Network | Images | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| ----------- | ------- | ------ | ----------- | ------------ | ------------ |
| PUCPR       | MiniVGG | 100    | 89.72%      | 49.41%       | 88.92%       |
| PUCPR       | MiniVGG | 200    | 99.95%      | 99.02%       | 95.77%       |
| PUCPR       | MiniVGG | 1000   | 99.99%      | 98.17%       | 96.25%       |

| Parking lot | Network | Epochs | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| ----------- | ------- | ------ | ----------- | ------------ | ------------ |
| ALL         | VGG16   | 5      | 93.20%      | 77.69%       | 98.87%       |
| ALL         | VGG16   | 10     | 98.08%      | 92.81%       | 96.76%       |
| ALL         | VGG16   | 15     | 97.48%      | 95.17%       | 98.05%       |

## Regularization

> Use PUCPR training set & MiniVGG network

| Id  | Method              | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| --- | ------------------- | ----------- | ------------ | ------------ |
| r0  | None                | 99.81%      | 92.55%       | 92.53%       |
| r1  | Batch Normalization | 99.94%      | 96.73%       | 91.02%       |
| r2  | L2 regularization   | 99.43%      | 91.27%       | 77.62%       |
| r3  | Dropout             | 99.95%      | 96.95%       | 92.84%       |
| r4  | BN + Dropout        | 99.95%      | 99.02%       | 95.77%       |
| r5  | BN + L2 + Dropout   | 99.87%      | 92.51%       | 94.45%       |

> Use PUCPR training set & VGG-13 network

| Id   | Method              | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| ---- | ------------------- | ----------- | ------------ | ------------ |
| r0   | None                | 00.00%      | 00.00%       | 00.00%       |
| r1   | Batch Normalization | 00.00%      | 00.00%       | 00.00%       |
| r2   | L2 regularization   | 00.00%      | 00.00%       | 00.00%       |
| r3   | Dropout             | 00.00%      | 00.00%       | 00.00%       |
| r4   | BN + Dropout        | 99.82%      | 94.05%       | 93.66%       |
| r5   | BN + L2 + Dropout   | 99.97%      | 98.37%       | 95.13%       |

## Verbose

### train_network.py

```bash
(Tensorflow-3.6) C:\Users\AWAKE\Desktop\GP\ips>python train_network.py -d train_data/train/ -m train_data/models/vgg16-200.model
Using TensorFlow backend.
[INFO] loading images...
[INFO] compiling model...
2019-03-30 14:43:39.127451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7845
pciBusID: 0000:01:00.0
totalMemory: 6.00GiB freeMemory: 4.96GiB
2019-03-30 14:43:39.135581: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-03-30 14:43:40.248390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-30 14:43:40.253854: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-03-30 14:43:40.256276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-03-30 14:43:40.260234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4714 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
[INFO] training network...
Epoch 1/5
819/819 [==============================] - 95s 116ms/step - loss: 0.2855 - acc: 0.9539 - val_loss: 0.1071 - val_acc: 0.9681
Epoch 2/5
819/819 [==============================] - 87s 107ms/step - loss: 0.0765 - acc: 0.9789 - val_loss: 7.0363 - val_acc: 0.5226
Epoch 3/5
819/819 [==============================] - 87s 107ms/step - loss: 0.1046 - acc: 0.9717 - val_loss: 0.0206 - val_acc: 0.9933
Epoch 4/5
819/819 [==============================] - 89s 108ms/step - loss: 0.0598 - acc: 0.9838 - val_loss: 0.0252 - val_acc: 0.9931
Epoch 5/5
819/819 [==============================] - 94s 115ms/step - loss: 0.0407 - acc: 0.9885 - val_loss: 0.2214 - val_acc: 0.9199
[INFO] serializing network...
```

### calc_accuracy.py

```bash
(Tensorflow-3.6) C:\Users\AWAKE\Desktop\GP\ips>python calc_accuracy.py
Using TensorFlow backend.
[INFO] loading network...
2019-03-31 20:00:34.046661: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7845
pciBusID: 0000:01:00.0
totalMemory: 6.00GiB freeMemory: 4.96GiB
2019-03-31 20:00:34.055607: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-03-31 20:00:35.094689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-31 20:00:35.100110: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-03-31 20:00:35.103378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-03-31 20:00:35.107615: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4714 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
100%|#######################################################################################################| 60000/60000 [05:02<00:00, 198.42it/s]
Accuracy of PUCPR: 98.87%
Accuracy of UFPR04: 95.65%
Accuracy of UFPR05: 86.81%
```

## References

- [Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)
- [Amazon Machine Learning](https://docs.aws.amazon.com/zh_cn/machine-learning/latest/dg/what-is-amazon-machine-learning.html)
