# Image Transformer
[![tensorflow](https://img.shields.io/badge/tensorflow-1.11-ed6c20.svg)](https://www.tensorflow.org/)


## Introduction

This TensorFlow implementation is designed with these goals:
- [ ] **Image Transformer** Tensorflow implementatin of [Image Transformer](https://arxiv.org/abs/1802.05751)
- [ ] Implementing New Ideas.
  - [ ] Local Sliding Self-Attension
  - [ ] Apply [Universal Transformers](https://arxiv.org/abs/1807.03819) concept

## How to Use
* cifar10
```
python3 bin/train_estimator_cifar10.py --model-dir models/test
```

### Prerequisite

Should install below libraries.

- Tensorflow >= 1.11
- opencv >= 3.0

And install below dependencies.

```bash
apt install -y libsm6 libxext-dev libxrender-dev libcap-dev
pip install -r requirements.txt
```

