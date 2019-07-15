<p><a href="https://www.vision-impulse.com"><img align="right" height="35%" width="35%" src="https://vision-impulse.com/wp-content/uploads/2019/06/logo_updated-e1560941558845.png" alt="Vision Impulse Logo"></a></p><br>
<p align="left">

# UrbanAI
**Mapping Human Settlements with Sentinel-2 Satellite Imagery**

![Sentinel-2 based Human Settlement Prediction](http://vision-impulse.com/vi-images-public/human_settlements.png)
<p align="center"><i><b>Figure 1.</b> Sentinel-2 based Human Settlement Prediction.</i></p>

Since 2015 the Sentinel-2 mission is sensing our Earth delivering multi-spectral satellite imagery with a spatial resolution of up to 10 meters per pixel. Each week multiple terabytes of multi-spectral satellite data covering the visible, near infrared, and short wave infrared part of the electromagnetic spectrum are collected. The data is freely and openly available for commercial and non-commercial purposes with the intention to fuel innovation. In order to make use of this large amount of Sentinel-2 satellite imagery, we provide an annotated machine learning datasets as well as a toolset to train Machine Learning, in particular, Deep Learning models with this Sentinel-2 data. The availability of such models allows exploiting the full potentialities of the Sentinel-2 satellites. The exploitation of the Sentinel-2 archive again allows to addressing pressing challenges and improving business decision making.


# About the Project

Half of humanity – 4.2 billion people – lives in cities today and 6.5 billion people are projected to live in cities by 2050 resulting in extreme transformations of land, in particular, in Africa and China experiencing the most rapid urban growth (e.g. in China, about 300 million rural inhabitants will move to urban areas over the next 15 years) [[1]](https://www.undp.org/content/undp/en/home/sustainable-development-goals/goal-11-sustainable-cities-and-communities.html). Rapid urbanization is exerting pressure on fresh water supplies, sewage, the living environment, and public health. Therefore, continuous monitoring of urban areas is getting more and more important. In this project, we show the effectiveness of AI methods applied to Sentinel-2 satellite imagery for the task of mapping human settlements. This project implements a Sentinel-2 image segmentation of urban areas by using the multi-spectral sensor information. Such EO-based urban classification products are critically needed at the global scale in order to help manage rapidly growing cities.


## Components 

* We release an urbanization dataset consisting of multi-spectral Sentinel-2 satellite images annotated with masks of human settlements obtained from the European Urban Atlas. The dataset covers 304 cities in Europe. We open source the code to download and expand the used dataset (e.g., to advanced/related LULC classification tasks).

* We release the code to train Deep Neural Networks with large EO data, in particular with Sentinel-2 satellite images. We created a network which can be trained with multi-spectral Sentinel-2 satellite images. The models rely on existing open-source technologies (**PyTorch** [[2]](https://github.com/pytorch/pytorch), **eo-learn** [[3]](https://github.com/sentinel-hub/eo-learn)), exploit cloud infrastructure (**Sentinel Hub** [[4]](https://github.com/sentinel-hub/sentinelhub-py)) and use accessible datasets (**Geopedia** [[5]](https://geopedia.world)).

* We provide a demonstrator showing the effectiveness of the trained models on unseen Sentinel-2 satellite images and show how to create light-weight models for embedded systems (Nvidia Jetson) with **ONNX** [[6]](https://onnx.ai/) and **NVIDIA TensorRT** [[7]](https://developer.nvidia.com/tensorrt).


## Data

In order to extract human footprint information, we use the European Urban Atlas covering hundreds of European cities. The UrbanAI dataset is created by annotating Sentinel-2 satellite images with this information. The created dataset contains human settlement annotations from 304 European cities. The dataset allows to train in particular supervised machine learning models to extract geospatial urban properties. It is easily accessible as open data in the EOPatch format. Please find the documentation about the EOPatch format here [[eolearn.core]](https://eo-learn.readthedocs.io/en/latest/examples/visualization/EOPatchVisualization.html?highlight=EOPatch#EOPatch-visualizations). The created dataset can be downloaded using the provided crawler script in the this repository. In the corresponding notebook, we describe how to use the data and how it was generated. **Figure 2** shows the distribution of the cities covered in the UrbanAI dataset.


<p align="center">
  <img width="50%" height="50%" src="http://vision-impulse.com/vi-images-public/urbanai_distribution.png" alt="Cities covered in the UrbanAI dataset.">
</p>

<p align="center">
  <i><b>Figure 2.</b> European cities covered in the UrbanAI dataset. Visualization source: Geopedia.</i>
</p>


The models are trained on Sentinel-2 satellite imagery. In this study, we consider the part of the electromagnetic spectrum covered by Sentinel-2. The dataset is created using the services  **Sentinel Hub** [[4]](https://github.com/sentinel-hub/sentinelhub-py) and **Geopedia** [[5]](https://geopedia.world). **Figure 3** shows the RGB bands of an EOPatch with the corresponding human settlement mask. 


<p align="center">
  <img width="50%" height="50%" src="http://vision-impulse.com/vi-images-public/vi_hs_example_prediction.png" alt="Sentinel-2 based Human Settlement Prediction">
</p>

<p align="center">
  <i><b>Figure 3.</b><b> Left:</b> Sentinel-2 image of a single EOPatch. <b>Right:</b> Corresponding human settlement annotations in yellow. </i>
</p>

*The Images are extracted from the S2 cloudless layer created with the sentinel-hub/sentinel2-cloud-detector. Sentinel Hub configuration: Mosaic order: Least cloud coverage, Cloud coverage: 5%.*


## Technology

### Semantic Segmentation


The developed solution uses the EO processing framework **eo-learn** [[3]](https://github.com/sentinel-hub/eo-learn) and the Deep Learning framework **PyTorch** [[2]](https://github.com/pytorch/pytorch). We use a U-Net architecture [[8]](https://arxiv.org/abs/1505.04597) for the following experiments (in addition we provide a plain FCN and a SegNet model). Figure 4 shows the network architecture. The U-Nets exploit the pre-trained encoders ResNet and MobileNetV2. To learn more about the encoders, please find the corresponding papers here:
 * ResNet [[9]](https://arxiv.org/abs/1512.03385)
 * MobileNetV2 [[10]](https://arxiv.org/abs/1801.04381)

<p align="center">
<img src="http://vision-impulse.com/vi-images-public/network_ms.png" alt="Multispectral Network Architecture" width="750">
</p>
<p align="center"><i><b>Figure 4.</b> U-Net architecture trained with input images with a dimension of (512, 512, X).</i></p>

The configuration files **config.py** and **jetson-model-config.py** allow to easily set training parameters for network optimization. Input images are normalized across each channel (per channel min-max normalization).

Example configuration:
```
EPOCHS=30
OPTIMIZER='adam'
MODALITY='MS'
LR=0.0001
LOSS='BCEJaccardLoss'
BATCH_SIZE=8
PRE_TRAINED=True
DATA_AUG=['RandomHorizontalFlip', 'RandomVerticalFlip']
```

#### Network Expansion and Transfer Learning

The standard U-Net architecture [[8]](https://arxiv.org/abs/1505.04597) is trained with three channel input images. We expand the three channel network to be trained with multispectral imagery. In order to achieve this, we have extended the input layer and changed the preprocessing and data augmentation procedures allowing the network to be trained with an arbitrary number of input channels **X**. In order to use pre-trained models for transfer learning, the additional input channels are initialized with the replicated weights of the pre-trained channels. Input images are normalized across each channel (per channel min-max normalization). The training routine is available in this repository.


#### Nvidia Jetson Platform

We created light-weight models allowing to run inference on embedded systems such as CubeSats or drones (e.g., to provide onboard processing for time-critical applications such as in the context of emergency response). We deployed the models on the Jetson platform - Nvidia's embedded computer vision platform providing GPU-accelerated data processing. The models use a ResNet [[9]](https://arxiv.org/abs/1512.03385) and a MobileNetV2 [[10]](https://arxiv.org/abs/1801.04381) encoder. Please find step-by-step instructions on how to set up an Nvidia Jetson device, how to export **PyTorch** [[2]](https://github.com/pytorch/pytorch) models in the appropriate format using **ONNX** [[6]](https://onnx.ai/) and **NVIDIA TensorRT** [[7]](https://developer.nvidia.com/tensorrt) as well as instructions how to deploy and run the inference models on the Nvidia Jetson platform in this repository.

## Demonstrator: Human Settlement Layer

We developed a demonstrator which allows using the trained models to map human settlements with Sentinel-2 satellite images. The Jupyter Notebook downloading the corresponding EOPatches is available in this repository. Please use eo-learn-mask to create the cloud mask for masking the prediction of the analyzed EOPatch. The predictions for the individual patches can be merged to get a georeferenced mapping for an entire city/country/continent. **Figure 5** shows a mapping of the settlements in and around the city of Tübingen, Germany.

*This visualization is based on a prototype. If you use the models in further scenarios, please keep in mind to add non-human settlement images during training and to create a dataset covering the intra-class variance of settlements at a global scale. Possible enhancements are global predictions, time series analysis or a detailed analysis (e.g., mapping of residential and industrial buildings or mapping the population density). If you are interested, please feel free to contact us via info[at]vison-impulse.com.*

<p align="center">
<img src="http://vision-impulse.com/vi-images-public/prediction_tuebingen.png" alt="Multispectral Network Architecture" width="450">
</p>
<p align="center"><i><b> Figure 5.</b> Human settlement prediction in and around the city of Tübingen, Germany.</i></p>

## The Authors

**UrbanAI** was implemented by Vision Impulse GmbH. The project is part of QueryPlanet funded by **ESA** [[11]](https://www.esa.int). The consortium in charge of implementing this project consists of **Sinergise** (Ljubljana, Slovenia) [[12]](https://www.sinergise.com), **Vision Impulse** (Kaiserslautern, Germany) [[14]](https://www.vision-impulse.com), and  **Development Seed** (Lisbon, Portugal) [[13]](https://developmentseed.org). The following Vision Impulse authors have contributed to the project:

* Patrick Helber
* Benjamin Bischke
* Nicolas Ventulett
* Michel Klomp
* Qiushi Guo

<p align="center">
  <a href='#'><img src="http://vision-impulse.com/vi-images-public/consortium_logos.png" alt="Logos Consortium"></a>
</p>

## Contact
E-Mail: info[at]vison-impulse.com

## License
This project is licensed under the terms of the [MIT license](LICENSE).
