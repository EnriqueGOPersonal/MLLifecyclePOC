#!/usr/bin/env python
# coding: utf-8

# # Credijusto Data Scientist Challenge 💻💰🚀
# 
# ## Dataset description.
# 
# #### 1) Personal [data table]
# - **client_id**
#     - key to job table
#     - key to bank table
#     - key to transactional data table
# - name
# - address
# - phone_number
# - email_domain
# - smoker
# - is_married
# - car_licence_plate
# - age
# - number_of_children
# - years_of_education
# - has_criminal_records
# 
# #### 2. Job [data table]
# - **client_id**
#     - key to personal table
#     - key to bank table
#     - key to transactional data table
# - company
# - phone_number
# - address
# - email_domain
# - current_job
# - car_licence_plate
# - years_in_current_job
# - salary
# 
# #### 3. Bank [data table]
# - **client_id**
#     - key to personal table
#     - key to job table
#     - key to transactional data table
# - account_id
#     - key to transactional data table
# - number_of_credit_cards
# - number_logs_per_day
# - number_secret_keys_requested
# - credit_card_number
# - credit_card_expire
# - credit_card_provider
# - credit_score
# - first_credit_card_application_date
# - last_credit_card_application_date
# - **defaulted_loan**
#     - Variable to predit
# 
# #### 4. Transactional [data table]
# - **transaction_id**
# - **account_id**
#     - key to bank table
# - **client_id**
#     - key to personal table
#     - key to job table
#     - key to bank data table
# - duration_minutes
# - amount
# - type
# - date
# 
# ## Business question
# 
# #### Background
# 
# 1. **Only the training set bank data table has the column defaulted_loan** which has two different outcomes:
#     - True
#         - Client defaulted (did not pay credit).
#         - This is the *Positive class*
#     - False
#         - Client is OK (did pay credit).
#         - This is the *Negative class*
# 2. You need to make a predictive model to **make predictions of the feature defaulted_loan on the test dataset**.
# 3. **The evaluation of this challenge relies only on the prediction scores on test dataset**.
#     - Choose wisely the evaluation metric for this challenge.
# 

# # Problem definition
# 
# Lenders provide loans to borrowers in exchange for the promise of repayment with interest. That means the lender only makes profit (interest) if the borrower pays off the loan. However, if he/she doesn’t repay the loan, then the lender loses money.
# 
# Therefore the lending industry is based in the answers of two critical questions: 
# 
# 1) How risky is the borrower?
# 
# 2) Given the borrower’s risk, should we lend him/her? 
# 
# The answer to the first question determines the interest rate the borrower would have. Interest rate measures among other things (such as time value of money) the riskness of the borrower, i.e. the riskier the borrower, the higher the interest rate. With interest rate in mind, we can then determine if the borrower is eligible for the loan.
# 
# "Predicting Loan Repayment", Imad Dabbura https://towardsdatascience.com/predicting-loan-repayment-5df4e0023e92 [1]
# 
# As stated in the Business question, for our purposes we would only predict the answer of the question 1.
# 
# ## The importance of predicting right the borrower's riskness
# 
# bla bla...
# 

# # Set working environment

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle, safe_indexing
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
import io
import matplotlib as plt
# Run this to install not common libraries
#!pip install eli5
import eli5


# # Importing data

# In[2]:


data = {
    'train': {
        'personal': pd.read_csv('data/client_personal_train.csv'),
        'job': pd.read_csv('data/client_job_train.csv'),
        'bank_data': pd.read_csv('data/client_bank_data_train.csv'),
        'transactional_data': pd.read_csv('data/client_transactional_data_train.csv')      
    },
    'test': {
        'personal': pd.read_csv('data/client_personal_test.csv'),
        'job': pd.read_csv('data/client_job_test.csv'),
        'bank_data': pd.read_csv('data/client_bank_data_test.csv'),
        'transactional_data': pd.read_csv('data/client_transactional_data_test.csv')
    }
}


# # Data exploration

# ## Train Data exploration

# ### 1 - **Checking datasets dimesions**

# In[3]:


[print('Dataset: ' + x + ' | Dataset dimension (rows, cols): ' + str(data['train'][x].shape)) for x in data['train'].keys()]


# ### 2 - **Checking Row example values**

# #### 1) Personal datatable

# In[4]:


data['train']['personal'].head()


# **Note**:
# * Although it is currently not possible, with the help of an hypothetical additional  "*common_names_dataset*" for female and male individuals, genre information can be extracted from _name_ column for future analysis.
# * Depending on quality of the information contained in the column _address_ , geographical data might be useful for future analysis. For simplicity we won't do it in this notebook.
# * *car_licence_plate*, *phone_number* and *email_domain* columns seem to have no useful information.

# #### 2) Job datatable

# In[5]:


data['train']['job'].head()


# **Note**:
# * Depending on quality of the information contained in the column _address_ , geographical data might be useful for future analysis.
# * *car_licence_plate*, *phone_number* and *email_domain* columns seem to have no useful information.

# #### 3) Bank datatable
# - Notice that this is the table that contains variable to predict: **defaulted_loan**

# In[6]:


data['train']['bank_data'].head()


# **Note**:
# * Depending on quality of the information contained in the column _address_ , geographical data might be useful for future analysis.
# * *credit_card_number*, *phone_number* and *email_domain* columns seem to have no useful information.

# #### 4) Bank transactions datatable

# In[7]:


data['train']['transactional_data'].head()


# **Note**:
# * 
# * *transaction_id* column seem to have no useful information.

# ### 3 - **Checking Data types**
# 

# In[8]:


[print("Datatypes for -" + x + "- dataset are:\n\n" + f"{data['train'][x].dtypes}\n", end = "\n") for x in data['train'].keys()]


# ### 4 - **Checking Null Values**

# In[9]:


[print("Dataset -" + x + "- contains the following number of null values by feature: \n\n" + f"{data['train'][x].isnull().sum()}\n", end = "\n") for x in data['train'].keys()]


# **Note**:
#     * No missing values found in the train set.

# ### 5 - **Checking number of repeated values in ID columns**
# 
# This is done to define what kind of merging to do later on the datasets and expect certain behavior.

# In[10]:


for dataset in data["train"]:
    df = data["train"][dataset]
    print("-"*100)
    print("For dataset " + dataset)
    for id_col in df.columns:
        if "_id" in id_col:
            num_uniques = len(df[id_col].unique())
            print("There are "+ str(num_uniques) + " distinct " + id_col)


# From the previous outputs we can conclude that:
# - There are no repeated clients in the personal dataset.
# - One client has exactly one job in the job dataset.
# - One client has exactly one account in the bank data dataset.
# - One client has 1 or more transactions in the transactional dataset.
# 
# **Note**:
# 
# In order to later use the dataset *transactional data* to make one single prediction per client, we have to group the data by client, engineering features from it.
# 
# 

# ### 6 - **Label stats**
# - **defaulted_loan**: if True, it means that the client defaulted the loan. If False, client paid the loan.
# - **Our interest is to predict if a credit applicant (client_id) will default the loan.**

# In[11]:


data['train']['bank_data']['defaulted_loan'].value_counts()


# In[12]:


100 * np.round(data['train']['bank_data']['defaulted_loan'].value_counts() / data['train']['bank_data'].shape[0], 2)


# **Note**:
# - Currently, only 5% of the portfilio has defaulted the loan. This indicates that we will have to later manage an unbalanced label dataset to get better expected results.

# ## Test set exploration

# In[13]:


[print('Dataset: ' + x + ' | Dataset dimension (rows, cols): ' + str(data['test'][x].shape)) for x in data['test'].keys()]


# In[14]:


[print("Dataset -" + x + "- contains the following number of null values by feature: \n\n" + f"{data['test'][x].isnull().sum()}\n", end = "\n") for x in data['test'].keys()]


# **Note**:
# - No missing values found in the train set.
# 
# For simplicity, we will assume:
# * Test set comes from the same distribution as train set (therefore the same distribution of dev set).
# * Test set does not contain values not contained in the train set for categorical one-hot encoded features (as this would require to handle this exceptions by replacing those values, droping rows with those values or redefining the feature encoding).

# # Data wrangling and feature extraction for Exploratory Data Analysis (EDA)

# The goal of this section is to preprocess data to make EDA reveal clearer patterns more easily.
# 
# In this section we preprocess train and test data the same way to later be able to use the same model to make predictions on both.
# 
# **Note**:
# 
# Feature extraction is only one of a series of iterative trial and error steps in the machine learning cycle. This is only a first approach.
# 
# ## Dropping not useful columns from datasets
# 
# As mentioned on the Train Data Exploration section, we may remove the following unuseful feature columns from datasets:
# * name
# * address
# * car_licence_plate
# * phone_number
# * email_domain 
# * credit_card_number

# In[15]:


columns_to_drop = ["car_licence_plate", "phone_number", "email_domain", "name", "address", 
                   "credit_card_number", "credit_card_number"]

for dataset in data["train"]:
    data["train"][dataset] =     data["train"][dataset].drop([columname for columname in data["train"][dataset].columns
                                 if columname in columns_to_drop], axis = 1)
    print("* " + dataset + " training dataset " + " columns:")
#     print(data["train"][dataset].columns.values, end = "\n\n") # Uncomment this line to validate if the operation is succesful

for dataset in data["test"]:
    data["test"][dataset] =     data["test"][dataset].drop([columname for columname in data["test"][dataset].columns 
                                 if columname in columns_to_drop], axis = 1)
#     print(data["test"][dataset].columns) # Uncomment this line to validate if the operation is succesful


# ## Converting date columns to datetime type
# 
# The following columns currently are of type "object", they should be of type "datetime":
# * credit_card_expire (from bank_data dataset)
# * first_credit_card_application_date (from bank_data dataset)
# * last_credit_card_application_date (from bank_data dataset)
# * date (from transactional_data dataset)

# In[16]:


for dataset in ["train", "test"]:
    data[dataset]["bank_data"].credit_card_expire = data[dataset]["bank_data"].credit_card_expire.apply(lambda x: pd.to_datetime(r"01/" +x ))
    data[dataset]["bank_data"].first_credit_card_application_date = data[dataset]["bank_data"].first_credit_card_application_date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    data[dataset]["bank_data"].last_credit_card_application_date = data[dataset]["bank_data"].last_credit_card_application_date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    data[dataset]["transactional_data"].date = data[dataset]["transactional_data"].date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")


# Showing data successful type transformation:

# In[17]:


data["train"]["bank_data"][["credit_card_expire", "first_credit_card_application_date", 
                            "last_credit_card_application_date"]].dtypes


# In[18]:


data["train"]["transactional_data"][["date"]].dtypes


# ## Generating *transactional_data* dataset grouped by _client_id_

# In[19]:


for dataset in ["train", "test"]:
    df = data[dataset]["transactional_data"]
    # Defining new dataset with a single column of unique client IDs
    data[dataset]["transactional_data_gr"] = pd.DataFrame(pd.Series(df.client_id.unique()), columns = ["client_id"])
    # Filtering transactional_data dataset for each transaction type to calculate aggregation metrics on it
    for transaction_type in df.type.unique():
        temp = df[df.type == transaction_type]
        # Calculating aggregation metrics
        temp = temp.groupby(["client_id"]).agg({"client_id": "count", "amount": ["mean", "sum"], "duration_minutes": "mean", "date": ["min", "max"]})
        temp = temp.rename(columns = {"client_id": "num_transactions"})
        temp.columns = [(transaction_type.lower() + "_" + col[0] + "_" + col[1]) for col in temp.columns] # Renaming columns by transaction type
        temp = temp.reset_index()
        data[dataset]["transactional_data_gr"] = data[dataset]["transactional_data_gr"].merge(temp, on = "client_id")
    # Making sure all client_id rows are unique
    assert len(data[dataset]["transactional_data_gr"]) == len(data[dataset]["transactional_data_gr"].client_id.unique())

print("The transactional_data_gr dataset contains the following columns generated by grouping by client_id: \n ")
print(data[dataset]["transactional_data_gr"].columns.values)


# The feature-engineered columns generated so far are for each transaction type:
# * number of transactions
# * mean and total sum of amounts
# * mean duration (in minutes)
# * max and min transaction dates
# 
# Showing the resulting grouped dataset:

# In[20]:


data["train"]["transactional_data_gr"].head()


# In[21]:


data["train"]["transactional_data_gr"].dtypes


# ## Merging datasets into single train and test dataframes

# In[22]:


for d_set in ["train", "test"]:    
    data[d_set]["merged_"+d_set] = data[d_set]['personal'].merge(data["train"]['job'], on = "client_id", how = "left")
    data[d_set]["merged_"+d_set] = data[d_set]["merged_"+d_set].merge(data[d_set]['bank_data'], on = "client_id", how = "left")
    data[d_set]["merged_"+d_set] = data[d_set]["merged_"+d_set].merge(data[d_set]['transactional_data_gr'], on = "client_id", how = "left")
    # Droping id columns except client_id
    data[d_set]["merged_"+d_set] = data[d_set]["merged_"+d_set].drop(["account_id"], axis = 1)

print("The final " + str(len(data["train"]["merged_train"].columns)) + " trainset columns are: \n\n " +   str(data["train"]["merged_train"].columns))


# In[23]:


data["train"]["merged_train"].head()


# ## Further feature engineering
# 
# We may perform further feature engineering by defining the next columns:
# 
# * transaction_days_range : Range of days between first and last transaction (float32).
# * first_transaction_month : Month of first transaction (str).
# * first_transaction_year : Year of first transaction (str).
# * monthly_avg_withdrawals : Monthly average withdrawal number (float32).
# * monthly_avg_deposits : Monthly average deposit number (float32).
# * monthly_avg_w_amount : Monthly average withdrawal amount (float32).
# * monthly_avg_d_amount : Monthly average deposit amount (float32).
# * first_cc_app_month : Month of first credit card application date (str).
# * first_cc_app_year : Year of first credit card application date (str).
# * cc_expire_month : Month of credit card expire date (str).
# * cc_expire_year : Year of credit card expire date (str).

# In[24]:


def feat_eng(df):
    df["first_transaction"] = df.apply(lambda x: max(x.withdrawal_date_min, x.deposit_date_min), axis = 1)
    df["last_transaction"] = df.apply(lambda x: min(x.withdrawal_date_max, x.deposit_date_max), axis = 1)
    df["transaction_days_range"] = df.apply(lambda x: (x.last_transaction - x.first_transaction).days, axis = 1)
    df["first_transaction_month"] = df.apply(lambda x: min(x.withdrawal_date_min, x.deposit_date_min).month, axis = 1)
    df["first_transaction_year"] = df.apply(lambda x: min(x.withdrawal_date_min, x.deposit_date_min).year, axis = 1)
    df["monthly_avg_withdrawals"] = df.withdrawal_num_transactions_count/(df.transaction_days_range.apply(lambda x: x/30))
    df["monthly_avg_deposits"] = df.deposit_num_transactions_count/(df.transaction_days_range.apply(lambda x: x/30))
    df["monthly_avg_w_amount"] = df.withdrawal_amount_sum/(df.transaction_days_range.apply(lambda x: x/30))
    df["monthly_avg_d_amount"] = df.deposit_amount_sum/(df.transaction_days_range.apply(lambda x: x/30))
    df["first_cc_app_month"] =  df.first_credit_card_application_date.apply(lambda x: x.month)
    df["first_cc_app_year"] :df.first_credit_card_application_date.apply(lambda x: x.year)
    df["cc_expire_month"] = df.credit_card_expire.apply(lambda x: x.month)
    df["cc_expire_year"] = df.credit_card_expire.apply(lambda x: x.year)
    return df

for d_set in ["train", "test"]:    
    data[d_set]["merged_"+d_set] = feat_eng(data[d_set]["merged_"+d_set])


# And droping the following (not useful anymore) columns:
# 
# * withdrawal_date_min
# * withdrawal_date_max
# * deposit_date_min
# * deposit_date_max
# * first_credit_card_application_date
# * last_credit_card_application_date
# * creditcard_expire

# In[25]:


columns_to_drop = ["withdrawal_date_min", 
"withdrawal_date_max",
"deposit_date_min",
"first_credit_card_application_date",
"last_credit_card_application_date",
"deposit_date_min", 
"deposit_date_max", 
"credit_card_expire"]

data["train"]["merged_train"] = data["train"]["merged_train"].drop(columns_to_drop, axis = 1)
data["test"]["merged_test"] = data["test"]["merged_test"].drop(columns_to_drop, axis = 1)


# We obtain the following resulting merged dataset:

# In[26]:


data["train"]["merged_train"].head()


# With column types:

# In[27]:


data["train"]["merged_train"].dtypes


# # Exploratory Data Analysis
# 
# ## Dividing features by data type
# 
# To generate better graphs, lets divide features into data into categorical, boolean and numerical data.

# In[28]:


eda_df = data["train"]["merged_train"].reset_index(drop = True)
cat_cols = []
bool_cols = []
num_cols = []

for column in eda_df.columns[eda_df.columns != "client_id"]:
    if data["train"]["merged_train"][column].dtypes in ["object"]:
        cat_cols.append(column)
    elif data["train"]["merged_train"][column].dtypes in ["bool"]:
        bool_cols.append(column)
    else:
        num_cols.append(column)


# ## Plotting categorical data

# Counting number of unique values for categorical data:

# In[29]:


for col in cat_cols:
    print(col)
    print(len(eda_df[col].unique()))


# ### Exploring company column
# 
# Company column seems to not be a candidate for one-hot encoding without preprocessing, as it would generate 49178 columns.
# 
# We can analyze the number of times a company appears in our training dataset to try grouping more relevant values (values for wich we may identify a clear pattern).

# In[30]:


company_df = eda_df.groupby("company", as_index = False).agg({"client_id": "count"})[["company", "client_id"]].rename(columns = {"client_id": "rows_count"})
company_df.sort_values("rows_count", ascending = False).head()


# In[31]:


company_df.describe()


# We can observe that at least 75% of the 49178 listed companies have a single row (client_id) in our dataset, and the company with most rows is Smith PLC (97 rows).
# 

# In[ ]:


companies_x_employees = company_df["rows_count"].value_counts()
companies_x_employees = companies_x_employees.reset_index().rename(columns = {"index": "num_employees", "rows_count": "num_companies"})
more_20_emp_companies = companies_x_employees[companies_x_employees.num_employees > 20]
print("By grouping all companies with 20 or more employees we are now left with: " + str(len(more_20_emp_companies)) + " companies")


# Let's replace the values from the company column of the companies with less than 20 employees with the string "Not Relevant" and plot it.

# In[ ]:


companies_to_replace = company_df[company_df.rows_count <= 20].company.unique()
eda_df[eda_df.company.isin(companies_to_replace)] = "Not Relevant"
# sns.countplot(x = "company", data = eda_df)
len(eda_df.company.unique())


# ### Exploring current_job column
# 
# Company column seems to not be a candidate for one-hot encoding without preprocessing, as it would generate 639 columns.
# 
# We can analyze the number of times a job appears in our training dataset to group the more relevant values (values for wich we may identify a clear pattern).

# ## Plotting barplots for categorical columns

# In[ ]:


for i in range(1):
    for column in cat_cols[0:1]:
        sns.countplot(data = eda_df, x = column, hue = "defaulted_loan")


# ## Plotting scatterplots between features

# In[ ]:


sns.pairplot(data["train"]["personal"].merge(data["train"]["bank_data"][["client_id", "defaulted_loan"]], 
                                             on = "client_id", how = "left"), hue="defaulted_loan", diag_kind="kde", s = 1)


# In[ ]:


sns.pairplot(data["train"]["job"].merge(data["train"]["bank_data"][["client_id", "defaulted_loan"]], 
                                             on = "client_id", how = "left"), hue="defaulted_loan", diag_kind="kde")


# ## Plotting pearson correlation matrix between features

# In[ ]:


# Calculate the correlation matrix
corr = data["train"]["merged_train"].corr()

# Plot the heatmap

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        linewidths=.5,
       cmap="RdBu_r", center = 0)


# # Feature extraction after Exploratory Data Analysis (EDA)
# 
# A well designed EDA may lead us to the creation of relevant features.
# 
# The goal of this section is to example the implementation of two features derived from EDA:
# * salary_per_children
# * salary_per_education_year

# In[ ]:


data["train"]["merged_train"].dtypes


# # Building and comparing models performance

# ## Defining data wrangling pipeline steps
# 
# ### 1) Null values Imputing
# 
# Theo need to do imputing as the trainset and testset do not contain missing values 
# 
# ### 2) Encoding categorical columns

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Separating categorical columns and numeric columns
bool_columns = []
categoric_columns = []
numeric_columns = []
date_columns = []
for x in data["train"]["merged_train"].columns:
    if data["train"]["merged_train"][x].dtypes in ["object"]:
        categoric_columns.append(x)
    elif data["train"]["merged_train"][x].dtypes in ["bool"]:
        bool_columns.append(x)
    elif data["train"]["merged_train"][x].dtypes in ["int64", "float64"]:
        numeric_columns.append(x)
    else:
        date_columns.append(x)
#Preprocessing date columns
for col in date_columns:
    data["train"]["merged_train"][col+"_year"] = data["train"]["merged_train"][col].dt.year
    data["train"]["merged_train"][col+"_month"] = data["train"]["merged_train"][col].dt.month
    data["train"]["merged_train"][col+"_day"] = data["train"]["merged_train"][col].dt.day
    data["train"]["merged_train"][col+"_quarter"] = data["train"]["merged_train"][col].dt.quarter
    data["train"]["merged_train"] = data["train"]["merged_train"].drop([col], axis = 1)

categoric_pipe = ColumnTransformer([
                                    ("ohe", OneHotEncoder(), categoric_columns)],
                                   remainder = "passthrough")
x = categoric_pipe.fit_transform(data["train"]["merged_train"])


# In[ ]:





# ## Defining train and dev sets
# 
# It is necessary to train the model my measuring it's performance on not seen data (usually called dev set). 
# We will assign 80% of the original train set data to our newly defined train set and the rest of 20% data to the dev set.

# In[ ]:


# Shuffle train set
from sklearn.model_selection import train_test_split
x, y = shuffle(data["train"]["merged_train"].drop(["defaulted_loan"], axis = 1), data["train"]["merged_train"]["defaulted_loan"])
# Assign 80% data to train set 20% data to dev set
x_train, x_dev, y_train, y_dev = train_test_split(x,
                                                  y,
                                                  test_size = 0.2)
from imblearn.oversampling import SMOTE


# ## Fitting pipelines to the dataset

# ### 1) Logistic Regression Model pipeline

# In[ ]:





# ## Random Forest Model

# # Error analysis
# 
# In this section we further explore the instances in which the model made a wrong prediction to try to find patters, generate model improval propositions and measure the time investment / reward ratio of each to take a decision of the next step to perform.

# # ELI5

# Bla... por que ELI5?

# # CSV output

# In[ ]:


demo_output.head()


# In[ ]:


# Export as CSV
demo_output.to_csv('growth_ds_challenge_luis_garcia.csv')


# # References:
# 
# The problem definition was heavily influenced by
# 
# [1] https://towardsdatascience.com/predicting-loan-repayment-5df4e0023e92
# 
