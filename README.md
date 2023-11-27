# Introduction

This project involves implementing a Gaussian Process Regression model from scratch, using Python,

and integrating it with FastAPI for deployment.

# Project Structure


```
.
├── data
│   ├── international-airline-passengers.csv
│   └── mauna_loa_atmospheric_co2.csv
|
├── output
│   |
│   ├── /all the images for the API Interfaces/
│   ├
│   ├── co2_concentration
│   │   └── /all the plots for gaussian process on co2 dataset in png format /
│   ├── air_passenger
│   │   ├── /all the plots for gaussian process on air passengers dataset in png format /
│   │   
│   └── evaluation.txt
│       
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── data
│   │   └── data_loader.py
│   ├── models
│   │   ├── GaussianProcess.py
│   │   └── kernels.py
│   └── templates
│   │   └── index.html
       	└── training.html
│   │   ├── prediction_international_air_passenger.html
│   │   └── predection_mauna_loa_atmospheric_co2.html
│   └── utils
│       └── utils.py
│   │   ├── evaluation.py
│   │   └── visualization.py
│   └── rest_api.py
│       
├── Dockerfile
├── README.md
└── requirments.txt

'''
data/: Contains the datasets used for training and testing.
output/: Contains all the plots of the process with the different kernals and also screen captures of the GUI of the api
src/: Houses the project source code, divided into data loading, model definition, templates, and utility functions.
rest_api.py: The script that orchestrates the training, prediction, and API setup.
main.py: The call and launcher of the API


### Gaussian Process Regression

1. Data Loading:
File: data_loader.py
Purpose: Load and preprocess the datasets for model training and testing.

Datasets:
international-airline-passengers.csv: Monthly international airline passengers data.
mauna_loa_atmospheric_co2.csv: Mauna Loa atmospheric CO2 concentration data.

2. Gaussian Process Model:
Files: GaussianProcess.py, kernels.py
Purpose: Implement Gaussian Process Regression and define various kernels for modeling : Gaussian kernal, RBF kernal, Rational Quadratic kernal, Sum kernal, product kernal.

3. initialize Training and Visualization and setting the rest api:
File: rest_api.py

Purpose: Train Gaussian Process models with different kernels and visualize the results to send them to html pages .

Steps:
Load and preprocess datasets.
Create instances of kernel classes (e.g., Gaussian, RBF, Rational Quadratic).
Create Gaussian Process instances with different kernels.
Fit the models on the datasets.
Visualize predictions and uncertainty using custom visualization functions.
Evaluate the models using custom evaluation functions.
Implement a REST API using FastAPI to make the Gaussian Process Regression accessible over HTTP.

Endpoints:
/train: Trains the Gaussian Process models on the provided datasets.
/predict: Placeholder for making predictions using the trained models.


4. running the Gaussian processor
File: main.py

5. Dockerization:
Files: Dockerfile, requirements.txt
Purpose: Containerize the application with Docker for seamless setup and deployment.
Steps:
Create a Dockerfile specifying dependencies and configurations.
Build a Docker image.
Run a Docker container.

### Results

Successfully trained Gaussian Process Regression models on two datasets.
Air_passengers dataset : the model gave a RMSE of ~113 due to the small size of the dataset so there still a room for improvment
Co2 concentration dataset: the model with the best kernal (Rational quadratic) gave a RMSE of ~15 whitch is great score

Integrated the models with FastAPI, providing HTTP access to training and prediction functionality.

Dockerized the application for easy deployment.


###  Instructions for Running the Application
1- Install dependencies: by running the following command

pip install -r requirements.txt

2- you can either start the application by running the main.py 
   then Open your browser and navigate to http://127.0.0.1:8000 to interact with and test the API.

or 
  you can build the application using docker 

	docker build -t your-image-name .

  then run it using the following docker command :

	docker run -p 8000:8000 your-image-name

