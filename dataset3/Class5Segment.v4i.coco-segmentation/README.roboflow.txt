
Class5Segment - v4 2023-05-03 6:29am
==============================

This dataset was exported via roboflow.com on May 2, 2023 at 11:35 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1500 images.
Clas are annotated in COCO Segmentation format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 256x256 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 51 percent of the image
* Random shear of between -13° to +13° horizontally and -21° to +21° vertically
* Random brigthness adjustment of between -18 and +18 percent
* Random exposure adjustment of between -13 and +13 percent
* Salt and pepper noise was applied to 2 percent of pixels


