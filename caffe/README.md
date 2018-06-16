# SAN
PyTorch implementation for [Partial Transfer Learning with Selective Adversarial Networks (CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Partial_Transfer_Learning_CVPR_2018_paper.pdf) 


## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

Python 2.7

## Modification on Caffe
We inherit the code from paper "Unsupervised Domain Adaptation by Backpropagation". We will introduce our modification on their codes. As for the differences of their codes from the original caffe, you can visit "https://github.com/ddtm/caffe/tree/grl" for details.

- Add "EntropyLoss" layer for entropy minimization loss.
- Add "AggregateWeight" layer to implement the weighted class weight and instance weight for our weighting mechanism. 

## Datasets
We use Office-31, Office-Caltech and ImageNet-Caltech dataset in our experiments. We use [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt), [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256) and [ImageNet-2012](http://www.image-net.org) datasets. The ImageNet-Caltech and ImageNet-Caltech dataset will be published soon. 

The lists of dataset are in [data](./data) directory. The "data/imagenet-caltech/imagenet_1000_list.txt" is too large, we put it on the [google drive](https://drive.google.com/open?id=1QARHJoxVpyB2EQZyrBbBHSiEQjBowPD2). 

For Office-31 dataset, "name_31_list.txt"(name="amazon", "webcam", "dslr") is the source list file and "name_10_list.txt" is the target list file.

You can also modify the list file(txt format) in ./data as you like. Each line in the list file follows the following format:
```
<image path><space><label representation>
```

## Compiling
The compiling process is the same as caffe. You can refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html).

## Training and Evaluation
First, you need to download the AlexNet pre-trained model on ImageNet from [here](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) and move it to [./models/bvlc_reference_caffenet](./models/bvlc_reference_caffenet).
Then, you can train the model for each dataset using the followling command.
```
dataset_name = office
./build/tools/caffe train -solver models/train/dataset_name/solver.prototxt -weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu gpu_id
```
You need to set the "test_iter" parameters in the solver file for each task. This parameter need to be set as the size of the target dataset for testing.

The accuracy is reported in the TEST phase of training.
