# Authors:
- Antoine Schutz
- Malak Lahlou
- Sara Anejjar

# Description:
The objective of this project is to predict the development of Cardiovascular Diseases (CVD) in a population of around 300 000 people. The data is provided by the [Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/annual_data/annual_2015.html) (BRFSS).

The data is composed of 321 features and 1 target variable. The target variable is a binary variable (1 or -1) that indicates whether an individual has a risk of developing CVD or not.

# Python files:
- `implementations.py` : contains the implementations of the 6 mandatory functions, as well as some helper functions to compute them.

- `cross_validation.py` : contains the cross validation functions for each model.

- `metrics.py` : contains the functions to compute the accuracy and f1 score of the models.

- `helpers.py` : contains helper functions to load the data and create the submission file.

- `Logistics.py` : contains implementation of the 2 mandatory functions related to Logistics regression

- `preprocessing.py` : contains the functions to preprocess the data : remove NaN, encode categorical values, perform feature expension.

- `run.ipynb` : contains the main runnable function. Performs the cross validation and output the csv file


# Run the code:

In order to run the code, you need to download the data to the `dataset` folder.

In order to generate the csv Submission file, run the first 4 cells (data importation and preprocessing) , and the last cell (# To create CSV , EXECUTE _/5) 
