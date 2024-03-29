# CAP-Detection

This project is far from finished. The Performance is not satisfactory.

Visualization of training. The MRI image to segment is taken from the test set. Every epoch the segmentation is saved as a image. 
...

## Possible further Approaches/Improvements

- Right now only a small portion of the dataset is used. [data/siemens_reduced_normalized](data/siemens_reduced_normalized) is the used dataset, the whole dataset can be found in [data/siemens_full](data/siemens_full). When training the network with the full dataset it will converge to classify everything as background. This is because many samples contain no class at all. This information where the classes are _not_ located is important however. For further improvements the loss function (Dice loss) propbably needs to be adjusted. [Tversky Focal Loss](https://arxiv.org/abs/1810.07842) could be an promising approach (I tried it a bit already but it does not work quite well yet).  
- The proposed SegNet is far from optimal. I would suggest trying the standard U-Net. It has proven to be very very good in biomedical image segmentation.
- The naive approach of [Early Fusion](https://medium.com/haileleol-tibebu/data-fusion-78e68e65b2d1) the images into one tensor is too simple for this problem. With early fusion the spatial information is lost after the first convolution layer. The [two-stream convolutional network approach](https://papers.nips.cc/paper/2014/file/00ec53c4682d36f5c4359f4ae7bd7ba1-Paper.pdf) could work far better. Has proven to work on "normal" videos quite well. The temporal network would need to be replaced by a "spatial conv net" and i would suggest to take the simple image difference instead of the optical flow since in the MRI images there is always much movement of pixels. 
