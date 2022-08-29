## Data Scientist Udacity Nanodegree - Capstone Project: Starbucks Capstone Challenge

### Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## 1. Introduction <a name="introduction"></a>

![0](/pics/pic_sb.jpg)   


### Project Overview

This is the final project of my Udacity Data Scientist Nanodegree. I chose the Starbuck's Capstone Challenge project because I'm interested in the experience of studying user demographic purchasing behavior across different offer types to determine which demographics respond best to which offer type.

The data consists of simulated data that mimics customer behavior in the Starbucks Rewards mobile app. Starbuck regularly sends customers offers through the mobile app. The offers contain either advertising or various discount options.

Identifying the top responding demographics by offer type requires combining transactional, demographic, and offer data.

#### Capstone Project Report Structure
```
- Section 1: Project Definition
| - Project Overview
| - Project Statement
| - Metrics

- Section 2: Analysis
| - Data Exploration
| - Data Visualization

- Section 3: Methodology
| - Data Preprocessing
| - Implementation
| - Refinement

- Section 4: Results
| - Model Evaluation and Validation
| - Justification

- Section 5: Conclusion
| - Reflection  
| - Improvement
```


## 2. Installation <a name="installation"></a>

Everything will run with a Python 3.x version.
It is necessary to install the following libraries to run the code in Anaconda. 

```
import os as os
import numpy as np
import pandas as pd
from pandas.plotting import table 
import math
import json
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')

# ML tools: Import the ML libraries required 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
```

## 3. Project Motivation<a name="motivation"></a>

For this project Starbucks and Udacity are providing 3 data sets:

Protfolio (offer) — Containing offer ids and meta data about each offer sent during a 30-day test period (duration, type, etc.)
Profile (demographic) — Containing demographic data of the rewards program for each customer.
Transcript (transaction records) — Containing event logs like records for transactions, offers received, offers viewed, and offers completed.

Facts:
* The users might not receive any offer in a specific time range.
* The users are receiving different kinds of offers.

Tasks:
1. Exploring the data by wrangling.
2. Data Analyse by Visualization of the data.
3. Preprocessing the data.
4. Implementation of the machine learning model.
5. Refinement of the machine learning model.
6. Evaluation of the machine learning model and justification of the results.
7. Reflection about the solution and potential improvements.


## 4. File Descriptions <a name="files"></a>

### File structure
```
- data
|- portfolio.json  # offer ids and meta data about each offer (duration, type, etc.)
|- profile.json  # demographic data for each customer
|- transcript.json # records for transactions, offers received, offers viewed, and offers completed
|- data.csv # merged data frame
|- data_clean.csv # cleaned data frame for machine learning model

- pics
|- pic_sb.jpg
|- pic1.png
|- pic2.png
|- plot_q1_age_hist.png
|- plot_g2_gender_pie.png
|- plot_q3_income_hist.png
|- plot_q4_gender_age_cluster.png
|- plot_q5_gender_income_cluster.png
|- plot_q6_offer_type_age_cluster.png
|- plot_q7_offer_type_gender.png
|- plot_q8.1_gender_event.png
|- plot_g8.2_gender_f_event_pie.png
|- plot_q9.1_age_cluster_event.png
|- plot_g9.2_age_cluster_f_event_pie.png

- metrics
|- plot_m1.1_metrics_table.png
|- plot_m1.2_metrics_bar.png
|- plot_m2_precision_recall_curve.png
|- plot_m3_KNeighborsC_precision_recall.png

- Starbucks_Capstone_notebook.ipynb # Data Analysis document (Jupyter Notebook)
- Starbucks_Capstone_notebook.html
- README.md # Readme file
```


## 5. Results<a name="results"></a>


MY Github repository: [https://github.com/SebastianHess/Capstone_Project_Starbucks_Challenge](https://github.com/SebastianHess/Capstone_Project_Starbucks_Challenge)



## 6. Licensing, Authors, Acknowledgements<a name="licensing"></a>

* Thanks to my employer for enabling me to take the Udacity Nanodegree and for supporting me.
* Thanks to Udacity for the amazing [Data Scientist Nanodegree Programm](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
