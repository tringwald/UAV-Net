Code and models for "UAV-Net: A Fast Aerial Vehicle Detector for Mobile Platforms".

## Quick start

Setup Caffe-SSD as described [here](https://github.com/weiliu89/caffe/tree/ssd). Make sure that the Caffe Python libs are on the `$PYTHONPATH`.
Then run the command below. This should create a file called `detect_result.jpg` in the current working directory. As `--model-path` you can pass any subdirectory of `models` that contains a prototxt and weight file.
The `image` folder contains 3 example images from the DLR 3K, VEDAI and UAVDT datasets as shown in the paper.

```
python2 draw_detections --model-path models/UAVDT/UAVNet_100_5x5_5boxes/ --image images/uavdt_img000097.jpg --gpu 0
```

## Citation
If you use our work, please consider citing:

```
@InProceedings{Ringwald_2019_CVPR_Workshops,
author = {Ringwald, Tobias and Sommer, Lars and Schumann, Arne and Beyerer, Jurgen and Stiefelhagen, Rainer},
title = {UAV-Net: A Fast Aerial Vehicle Detector for Mobile Platforms},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
} 
```