# Anomaly Detector

## Project: OMSA Practicum Fall 2022 Project for Fortiphyd Logic: https://www.fortiphyd.com/
Team: Krishna Kumar (kkumar80@gatech.edu), Sumit Machwe (smachwe3@gatech.edu)

# Project Structure

* **Analysis** : Contains csv files representing summary and detail level performance metrics of individual models and model groups. 
* **EDA** :  Please refer to Jupyter Notebook `SWAT_Analysis_Full_Data_EDA.ipynb` for Exploratory Data Analysis.
  * _Plots_ : Contains plots for: 
    * heatmap across all features
    * Feature (continuous variables) data distribution.
    * Normal data lineplots by sub system and aggregated over 1 minute and 15 minute time interval (EDA).
    * Attack data lineplots by sub system and aggregated over 1 minute and 15 minute time interval (EDA).
* **PredictionPlots**
  * `% of identified attack` and `propensity of attacks` for individual models across all attackIds.
  * `% of identified attack` and `propensity of attacks` for model groups across all attackIds.

* **notebook** _(Note: These notebooks are for SWaT Dec2015 data)_
  * `NormalAndAttackDataPickling.ipynb`: This notebook reads raw data (normal and attack). Does data cleansing and persist `pickle` file for further analysis and modeling.
  * `anomaly_detection_SWaT_Dec2015_sigma_0.5.ipynb` and `anomaly_detection_SWaT_Dec2015_sigma_1_0.ipynb`: This is main notebook for Anomaly Detection and does the following: 
    * Read Normal and attack pickle data.
    * Prepare the data for modeling by partition across sub-systems.
    * Train models using `pycaret` across all 6 sub-systems.
    * Predict and plot anaomalies across all sub-systems using threshold standard deviation of 0.5 and 1.0 respectively.
    * Persist csv files with performance metrics for analysis of models.
