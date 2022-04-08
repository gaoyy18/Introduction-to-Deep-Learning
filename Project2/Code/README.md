# Sartorius - Cell Instance Segmentation

Detect single neuronal cells in microscopy images.
Code for the kaggle competition, see [here](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview).

## Prepare the dataset

1. Visualize Dataset. You can see masks of a single picture.
   Please refer to Visual.ipynb.
2. Trans CSV to COCO's json file.
   Please refer to Trans to COCO.ipynb.
3. Visual COCO's json file.You can see masks of a single picture using COCO API.
   Please refer to COCO test.ipynb.
4. Divide train dataset.
   4.1. Divide Pictures
   Please refer to Divide Pic.ipynb.
   4.2. Divide JSON
   Please refer to Divide COCO.ipynb.
   (note:json file still using the original train folder as path)

## Mask R-CNN implementation

1. Requirements
   ​	you can use the requirements.txt file to configure the environment:

   ~~~shell
   pip install -r requirements.txt
   ~~~
2. Train
   Please refer to `train.ipynb`. You can modify the training hyperparameter settings and methods.
3. Test
   Please refer to `test.ipynb`.

## UNet implementation

1. Requirements

​	you can use the requirements.txt file to configure the environment:

~~~shell
 pip install -r requirements.txt
~~~

2. Train and Test

  Details is provided in the `unet.ipynb`.

## Infer on the Kaggle
1. Infer Mask R-CNN model
   Please refer to inference-and-submission.ipynb.
   Our Kaggle Page:https://www.kaggle.com/tianshuo42/inference-and-submission
2. Infer Unet model
   Please refer to inference-unet.ipynb.
   Our Kaggle Page:https://www.kaggle.com/tianshuo42/inference-unet

