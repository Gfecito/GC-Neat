# GC-Neat, a Geometric and Context Network.

## Abstract


## Data
I used Scene Flow's FlyingThings3D: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
I specifically used the RGB images (cleanpass) and Disparity.

## On pre-processing
Since this is a relatively small scale project, I scaled down the images and the disparities using the scripts in
python/data_resizing.py


## Sources
End-to-End Learning of Geometry and Context for Deep Stereo Regression; Kendall et al.: 
https://openaccess.thecvf.com/content_ICCV_2017/papers/Kendall_End-To-End_Learning_of_ICCV_2017_paper.pdf

Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches; Zbontar and LeCun: 
https://arxiv.org/pdf/1510.05970.pdf