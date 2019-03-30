# Analysis of Parking Spaces

## Structure

```bash
.
├── network/
|   ├── lenet.py
|   └── tinyvgg.py
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

> All accuracy tests are based on 20,000 images

| Parking lot | Network | Images | Epochs | Acc (PUCPR) | Acc (UFPR04) | Acc (UFPR05) |
| ----------- | ------- | ------ | ------ | ----------- | ------------ | ------------ |
| PUCPR       | tinyVGG | 100    | 5      | 89.72%      | 49.41%       | 88.92%       |
| PUCPR       | tinyVGG | 200    | 5      | 99.92%      | 98.80%       | 90.10%       |
| PUCPR       | tinyVGG | 1000   | 5      | 99.99%      | 98.17%       | 96.25%       |
| PUCPR       | LeNet   | 200    | 5      | 99.84%      | 97.14%       | 84.74%       |
| PUCPR       | VGG16   | 200    | 5      | 99.76%      | 96.73%       | 91.59%       |
| ALL         | tinyVGG | 200\*3 | 5      | 99.74%      | 98.55%       | 99.61%       |
| ALL         | VGG16   | 200\*3 | 5      | 93.11%      | 78.83%       | 98.71%       |

## Verbose

VGG16

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

tinyVGG

```bash
(Tensorflow-3.6) C:\Users\AWAKE\Desktop\GP\ips>python train_network.py -d train_data/train/ -m train_data/models/tinyvgg-200.model
Using TensorFlow backend.
[INFO] loading images...
[INFO] compiling model...
2019-03-30 02:43:04.605660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7845
pciBusID: 0000:01:00.0
totalMemory: 6.00GiB freeMemory: 4.96GiB
2019-03-30 02:43:04.615508: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-03-30 02:43:05.730187: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-30 02:43:05.734832: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-03-30 02:43:05.737688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-03-30 02:43:05.742209: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4714 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
[INFO] training network...
Epoch 1/5
819/819 [==============================] - 26s 32ms/step - loss: 0.1253 - acc: 0.9609 - val_loss: 0.0443 - val_acc: 0.9866
Epoch 2/5
819/819 [==============================] - 20s 25ms/step - loss: 0.0476 - acc: 0.9843 - val_loss: 0.0286 - val_acc: 0.9910
Epoch 3/5
819/819 [==============================] - 26s 32ms/step - loss: 0.0406 - acc: 0.9868 - val_loss: 0.0358 - val_acc: 0.9896
Epoch 4/5
819/819 [==============================] - 21s 26ms/step - loss: 0.0357 - acc: 0.9896 - val_loss: 0.0249 - val_acc: 0.9915
Epoch 5/5
819/819 [==============================] - 19s 24ms/step - loss: 0.0300 - acc: 0.9912 - val_loss: 0.0139 - val_acc: 0.9963
[INFO] serializing network...
```

## References

[Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)
[Amazon Machine Learning](https://docs.aws.amazon.com/zh_cn/machine-learning/latest/dg/what-is-amazon-machine-learning.html)
