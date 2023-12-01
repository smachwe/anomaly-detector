# Anomaly Detector

## Project: OMSA Practicum Fall 2022 Project for Fortiphyd Logic: https://www.fortiphyd.com/
Team: Krishna Kumar (kkumar80@gatech.edu), Sumit Machwe (smachwe3@gatech.edu)

### **Please review the detailed report here: [Fortiphyd Logic - Anomaly Detection - Final Report.pdf](reports/Fortiphyd%20Logic%20-%20Anomaly%20Detection%20-%20Final%20Report.pdf)**

## Introduction
Goal of this project is to devise generalizable anomaly detection in industrial control system sensor data. Fortiphyd Logic is a cybersecurity startup firm that specializes in fortifying industrial networks from malicious cyber attacks.
According to the paper1 written by David Fromby (Fortiphyd Logic) most of the industrial IoT device infrastructure (programmable control logic devices) is vulnerable to cyber-attacks and may result in massive economic losses.
Early detection of such attacks can not only save critical time to recovery but also encourage the PLC manufacturing companies to implement better security controls around the devices and prevent value loss.

For this project, we analyze the sensor data from a scaled-down version of an industrial water treatment plant. The plant consists of six stage filtration process. Water was continuously run for 11 days through the system. Network and physical attacks were launched during this period. The data consists of physical properties related to the plant and treatment process as well as network traffic data. Data files consist of non-anomalous and anomalous data. Please refer to this paper for further details.2

## Approach
Developing a generic anomaly detection process is hard. Unfortunately, one model fits all does not apply, especially when we do not want to incorporate domain knowledge about the dataset. We can however develop a framework that can be applied to other use cases with reasonably minimal changes. The approach we took required not to delve into the physics (or chemistry) of the dataset, but rather identify dependent variables that can be used to describe normal state behavior over time and which can be used in building unsupervised learning model.
In pursuit of such an approach, we discovered ‘pycaret’ library which organizes various Anomaly Detection algorithms. Under the hood, it uses ‘PyOD’ open-source library.

**Please review the detailed report here: [Fortiphyd Logic - Anomaly Detection - Final Report.pdf](reports/Fortiphyd%20Logic%20-%20Anomaly%20Detection%20-%20Final%20Report.pdf)**

# Project Structure

* **Analysis** : Contains csv files representing summary and detail level performance metrics of individual models and model groups. 
* **EDA** :  Please refer to Jupyter Notebook [SWAT_Analysis_Full_Data_EDA.ipynb](EDA/SWAT_Analysis_Full_Data_EDA.ipynb) for Exploratory Data Analysis.
  * _Plots_ : Contains plots for: 
    * heatmap across all features
    * Feature (continuous variables) data distribution.
    * Normal data lineplots by sub system and aggregated over 1 minute and 15 minute time interval (EDA).
    * Attack data lineplots by sub system and aggregated over 1 minute and 15 minute time interval (EDA).
* **PredictionPlots**
  * `% of identified attack` and `propensity of attacks` for individual models across all attackIds.
  * `% of identified attack` and `propensity of attacks` for model groups across all attackIds.

* **notebook** _(Note: These notebooks are for SWaT Dec2015 data)_
  * [Normal and Attack Data Pickling.ipynb](notebook/NormalAndAttackDataPickling.ipynb) : This notebook reads raw data (normal and attack). Does data cleansing and persist `pickle` file for further analysis and modeling.
  * [anomaly_detection_SWaT_Dec2015_sigma_0.5.ipynb](notebook/anomaly_detection_SWaT_Dec2015_sigma_0.5.ipynb) and [anomaly_detection_SWaT_Dec2015_sigma_1_0.ipynb](notebook/anomaly_detection_SWaT_Dec2015_sigma_1_0.ipynb) : These are the main notebooks for Anomaly Detection and does the following: 
    * Read Normal and attack pickle data.
    * Prepare the data for modeling by partition across sub-systems.
    * Train models using `pycaret` across all 6 sub-systems.
    * Predict and plot anaomalies across all sub-systems using threshold standard deviation of 0.5 and 1.0 respectively.
    * Persist csv files with performance metrics for analysis of models.
