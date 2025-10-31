<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://raw.githubusercontent.com/ICAERUS-EU/.github/refs/heads/main/profile/ICAERUS_transparent.png"></a>
    <h3 align="center">FAMMS Project - Forest Aerial Monitoring and Management System </h3>
  </p>
</p>
</div>

## Table Of Contents
- [Summary](#summary)
- [Structure](#structure)
- [Models](#models)
- [Application](#application)
- [Authors](#authors)

## Summary

The FAMMS (Forest and Agricultural Monitoring and Management System) project aims to combat illegal logging and promote sustainable forestry practices by integrating advanced technologies such as drones and AI.

We propose an use of UAV images and videos to automatically recognize and count animals in extensive areas. 

Two image analysis and processing models are proposed in this repository:
- A Change Detection CNN model
- A Pattern Recognition Forest Classifier model

More details on models section.

## Structure

The repository folders are structured as follow:

- **data**: some example images to use application scripts and other images about used equipment. 
- **models:** models developed for change detection and pattern recognition.
- **appp:** MVP application with user dashboard and other features.
- **platform.json:** organized information about the models.



## Models

The [models](https://github.com/ICAERUS-EU/OCT_FAMMS_PROJECT/tree/main/models) developed are the following:

#### _[Change Detection CNN](https://github.com/ICAERUS-EU/OCT_FAMMS_PROJECT/tree/main/models/model_1)_
A convolutional neural network model trained on pairs of drone images (T₀/T₁) to detect canopy loss or disturbance.

#### _[Pattern Recognition Forest Classifier](https://github.com/ICAERUS-EU/OCT_FAMMS_PROJECT/tree/main/models/model_2)_
Random Forest model used for anomaly detection based on vegetation index (NDVI) and canopy texture.



## Authors
- FAMMS Project Team