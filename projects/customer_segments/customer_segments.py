"""
Machine Learning Engineer Nanodegree
Unsuperviced Learning
Project: Creating Customer Segments

Welcom to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been
provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project.
Sections that being with 'Implementation' in the header indicate that the following block of code will require additonal functionality
which you must provide. Instructions will be provided for each section and the specifics of the implentation are marked in the code
block with 'TODO' statement. Please be sure to read teh isntrctions carefull!

In addition to implementing code, there will be questions what you must answer which relate to teh proejct and your implementation.
Each section where you will answer a question is precded by a 'Question X' header. Carefully read each quedstion and provide
thorough answers in the following text boxes that being with 'Answer:'. Your proejct submission will be evalueated based on your answers
to each of the questions and the implemetatino you provide.
"""

"""
Getting Started

In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in monetary units) of
divers product categories for internal structure. One goal of this prjoect is to best describe the variation in the different types of
customers that a whoelsale distrubutor inteacts with. Doing so would equip the distributor with insight into how to best structure their
delivery service to meet the needs of each customer

The dataset for this proejct can be found on teh UCI Machine Learnign Repository. For the purposes of this project, teh features 'Channel'
and 'Region' will be excluded in the analysis - with focus instead on the six product categories recodrded for customers.

Run the code block below to load the wholesale customer dataset, along with a few of the necesary Python libraries reuired for this
project. You will konw the dataset loaded successufy if the size of the dataset is reported.
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
# %matplotlib inline

# Load the wholesale cusotmers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers data has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

"""
Data Exploration
In this section, you will begin exploring the data througth visualizxations and code to understand how each feature is related to the oterhs.
You will observe a statistical descirption of the dataset, consider the relevance of each feature, and select a few sample data points from
the dataset which you will track through the course of this preojct.
Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product
categories: 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', and 'Delicatessen'. Consider what each category represents in
terms of products you could purchase.
"""

# Display a descript of the dataset
display(data.describe())

"""
Anil's Analysis
There are 6 different categories of products.
All of them have 440 entries.
The average a customer spends on Milk is 5796.27.
And so one.
"""

"""
Implementation: Selecting Samples
To get a better understanding of the customers and how their data will transfrom through the analysis, it would be best to select a few
sample data points and explore them in more detail. In the code block below, add three indices of your choice to the indices list which
will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from
one another.
"""

from random import randint
# TODO: Select three indices fo your choice you wish to sample from the dataset
# indices = [100, 200, 300]
# Randomly picking
# indices = [randint(0, 440) for _ in range(30)]

# Entry 75 is interesting. It has only 3 money spent on Grocery and Detergents_Paper
# Entry 181 is interesting. It has 112151 spent on Fresh.
# Entry 269 is interesting. It generally has high numbers for Milk, Grocery and Detergents_paper

# print(data.loc[indices])
# print(data.keys())

indices = [75, 181, 269]
# Create a DataFrame of the chose samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesamle customers dataset:"
display(samples)

"""
Question 1
Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.
What kind of establishment (customer) could each of the three samples you've chosen represent?
Hint: Examples of establishments include places like markets, cafes, and retailers, among may others. Avoid using names for
establishments, such as saying "McDonalds" when describing a sample customer as a restaurant.

Answer:
Customer 0: This customer doesn't have Grocery or Detergents_Paper. But, definitley has Fresh, Milk and Frozen.

Customer 1: This customer has very high volume of "Fresh" purchase. So, it has to be a Grocery store. And I do see big numbers on Milk, Grocery
Frozen sections also. The Detergents_Paper and Deli is also reasonably high. I think, it is a Grocery store.

Customer 2:
"""

"""
Implementation: Feature Relevance
One interesting thought to consider is if one (or more) of this sex product categories is actually relevant for understanding customer
purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will
necessarily purchase some proportional amoutn of anohter category of products? We can make si determination quite easily by
training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can
predict the removed feature.
In the code block below, you will need to implent the following:
* Assigne new_data a copy of hte data by remobing a feature of your choice using the DataFrame.drop function.
* Use sklean. corss_validation.train_test_split to split the dataset into training and testing sets.
  * Use the removed feature as your target label. Set a test_size of 0.25 and set a random_state.
  * Import a decision tree regressor, set a random_state, and fit the learner to the training data.
* Report the prediction score of the testing set using the regressor's score function.
"""

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop(['Milk'], axis = 1)
print "new_data has {} samples with {} features each.".format(*new_data.shape)
display(new_data.describe())
new_label = data['Milk']
display(new_label.describe())

# TODO:  Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, test_size = 0.25, random_state = 100)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 100)
regressor.fit(X_train, y_train)

# TODO: Report the score fo the prediction using the testing set
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, regressor.predict(X_test))
print "Score on test set is {}".format(score)

# Produce a scatter matrix for each paire of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')