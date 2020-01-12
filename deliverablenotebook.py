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

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
sns.set(style="ticks")
import io
import matplotlib.pyplot as plt
# Un comment the following lines and run to install not common libraries
# !pip install eli5
# !pip install imblearn
import eli5
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from imblearn.over_sampling import SMOTE

# # Importing data

# In[3]:


data = {
    'train': {
        'personal': pd.read_csv(r'data/client_personal_train.csv'),
        'job': pd.read_csv(r'data/client_job_train.csv'),
        'bank_data': pd.read_csv(r'data/client_bank_data_train.csv'),
        'transactional_data': pd.read_csv(r'data/client_transactional_data_train.csv')      
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

# In[4]:


[print('Dataset: ' + x + ' | Dataset dimension (rows, cols): ' + str(data['train'][x].shape)) for x in data['train'].keys()]


# ### 2 - **Checking Row example values**

# #### 1) Personal datatable

# In[5]:


data['train']['personal'].head()


# **Note**:
# * Although it is currently not possible, with the help of an hypothetical additional  "*common_names_dataset*" for female and male individuals, genre information can be extracted from _name_ column for future analysis.
# * Depending on quality of the information contained in the column _address_ , geographical data might be useful for future analysis. For simplicity we won't do it in this notebook.
# * *car_licence_plate*, *phone_number* and *email_domain* columns seem to have no useful information.

# #### 2) Job datatable

# In[6]:


data['train']['job'].head()


# **Note**:
# * Depending on quality of the information contained in the column _address_ , geographical data might be useful for future analysis.
# * *car_licence_plate*, *phone_number* and *email_domain* columns seem to have no useful information.

# #### 3) Bank datatable
# - Notice that this is the table that contains variable to predict: **defaulted_loan**

# In[7]:


data['train']['bank_data'].head()


# **Note**:
# * Depending on quality of the information contained in the column _address_ , geographical data might be useful for future analysis.
# * *credit_card_number*, *phone_number* and *email_domain* columns seem to have no useful information.

# In[8]:


data['train']['bank_data'].describe()


# #### 4) Bank transactions datatable

# In[8]:


data['train']['transactional_data'].head()


# **Note**:
# * 
# * *transaction_id* column seem to have no useful information.

# In[8]:


data['train']['transactional_data'].describe()


# ### 3 - **Checking Data types**
# 

# In[9]:


[print("Datatypes for -" + x + "- dataset are:\n\n" + f"{data['train'][x].dtypes}\n", end = "\n") for x in data['train'].keys()]


# ### 4 - **Checking Null Values**

# In[10]:


[print("Dataset -" + x + "- contains the following number of null values by feature: \n\n" + f"{data['train'][x].isnull().sum()}\n", end = "\n") for x in data['train'].keys()]


# **Note**:
#     * No missing values found in the train set.

# ### 5 - **Checking number of repeated values in ID columns**
# 
# This is done to define what kind of merging to do later on the datasets and expect certain behavior.

# In[11]:


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

# In[12]:


data['train']['bank_data']['defaulted_loan'].value_counts()


# In[13]:


100 * np.round(data['train']['bank_data']['defaulted_loan'].value_counts() / data['train']['bank_data'].shape[0], 2)


# **Note**:
# - Currently, only 5% of the portfilio has defaulted the loan. This indicates that we will have to later manage an unbalanced label dataset to get better expected results.

# ## Test set exploration

# In[14]:


[print('Dataset: ' + x + ' | Dataset dimension (rows, cols): ' + str(data['test'][x].shape)) for x in data['test'].keys()]


# In[15]:


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

# In[16]:


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

# In[17]:


for dataset in ["train", "test"]:
    data[dataset]["bank_data"].credit_card_expire = data[dataset]["bank_data"].credit_card_expire.apply(lambda x: pd.to_datetime(r"01/" +x ))
    data[dataset]["bank_data"].first_credit_card_application_date = data[dataset]["bank_data"].first_credit_card_application_date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    data[dataset]["bank_data"].last_credit_card_application_date = data[dataset]["bank_data"].last_credit_card_application_date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    data[dataset]["transactional_data"].date = data[dataset]["transactional_data"].date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")


# Showing data successful type transformation:

# In[18]:


data["train"]["bank_data"][["credit_card_expire", "first_credit_card_application_date", 
                            "last_credit_card_application_date"]].dtypes


# In[19]:


data["train"]["transactional_data"][["date"]].dtypes


# ## Generating *transactional_data* dataset grouped by _client_id_

# In[20]:


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

# In[21]:


data["train"]["transactional_data_gr"].head()


# In[22]:


data["train"]["transactional_data_gr"].dtypes


# ## Merging datasets into single train and test dataframes

# In[23]:


for d_set in ["train", "test"]:    
    data[d_set]["merged_"+d_set] = data[d_set]['personal'].merge(data["train"]['job'], on = "client_id", how = "left")
    data[d_set]["merged_"+d_set] = data[d_set]["merged_"+d_set].merge(data[d_set]['bank_data'], on = "client_id", how = "left")
    data[d_set]["merged_"+d_set] = data[d_set]["merged_"+d_set].merge(data[d_set]['transactional_data_gr'], on = "client_id", how = "left")
    # Droping id columns except client_id
    data[d_set]["merged_"+d_set] = data[d_set]["merged_"+d_set].drop(["account_id"], axis = 1)

print("The final " + str(len(data["train"]["merged_train"].columns)) + " trainset columns are: \n\n " +   str(data["train"]["merged_train"].columns))


# In[24]:


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

# In[ ]:


def feat_eng(df):
    df["first_transaction"] = df.apply(lambda x: max(x.withdrawal_date_min, x.deposit_date_min), axis = 1)
    df["last_transaction"] = df.apply(lambda x: min(x.withdrawal_date_max, x.deposit_date_max), axis = 1)
    df["transaction_days_range"] = df.apply(lambda x: (x.last_transaction - x.first_transaction).days, axis = 1)
    df["transaction_days_range"] = df.apply(lambda x: max(x.transaction_days_range, 1), axis = 1)
    df["first_transaction_month"] = df.apply(lambda x: str(min(x.withdrawal_date_min, x.deposit_date_min).month), axis = 1)
    df["first_transaction_year"] = df.apply(lambda x: str(min(x.withdrawal_date_min, x.deposit_date_min).year), axis = 1)
    df["monthly_avg_withdrawals"] = df.withdrawal_num_transactions_count/(df.transaction_days_range.apply(lambda x: x/30))
    df["monthly_avg_deposits"] = df.deposit_num_transactions_count/(df.transaction_days_range.apply(lambda x: x/30))
    df["monthly_avg_w_amount"] = df.withdrawal_amount_sum/(df.transaction_days_range.apply(lambda x: x/30))
    df["monthly_avg_d_amount"] = df.deposit_amount_sum/(df.transaction_days_range.apply(lambda x: x/30))
    df["first_cc_app_month"] =  df.first_credit_card_application_date.apply(lambda x: str(x.month))
    df["first_cc_app_year"] :df.first_credit_card_application_date.apply(lambda x: str(x.year))
    df["cc_expire_month"] = df.credit_card_expire.apply(lambda x: str(x.month))
    df["cc_expire_year"] = df.credit_card_expire.apply(lambda x: str(x.year))
    return df

for d_set in ["train", "test"]:    
    data[d_set]["merged_"+d_set] = feat_eng(data[d_set]["merged_"+d_set])


# And droping the following (not useful anymore) columns:
# 
# * first_transaction
# * last_transaction
# * withdrawal_date_min
# * withdrawal_date_max
# * deposit_date_min
# * deposit_date_max
# * first_credit_card_application_date
# * last_credit_card_application_date
# * credit_card_expire

# In[ ]:


columns_to_drop = ["first_transaction",
                   "last_transaction", 
                   "withdrawal_date_min", 
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

# In[ ]:


data["train"]["merged_train"].head()


# With column types:

# In[ ]:


data["train"]["merged_train"].dtypes


# # Exploratory Data Analysis

# ## Dividing features by data type
# 
# To generate better graphs, lets divide features into data into categorical, boolean and numerical data.

# In[ ]:


eda_df = data["train"]["merged_train"].reset_index(drop = True)
cat_cols = []
bool_cols = []
num_cols = []
date_cols = []

for x in data["train"]["merged_train"].columns:
    if data["train"]["merged_train"][x].dtypes in ["object"]:
        cat_cols.append(x)
    elif data["train"]["merged_train"][x].dtypes in ["bool"]:
        bool_cols.append(x)
    elif data["train"]["merged_train"][x].dtypes in ["int64", "float64"]:
        num_cols.append(x)
    else:
        date_cols.append(x)


# ## Exploring categorical data

# Counting number of unique values for categorical data:

# In[66]:


for col in cat_cols:
    if col != "client_id":
        print("Column name: " + col)
        print("Number of unique values: " + str(len(eda_df[col].unique())))


# ### Exploring *company* column
# 
# Company column seems to not be a candidate for one-hot encoding without preprocessing, as it would generate 49178 columns and it is very unlikely that all of them have enough rows to generalize a pattern.
# 
# We can analyze the number of times a company appears in our training dataset to try to collapse categorical variables values into more relevant values (values for wich we may identify a clear pattern).
# 
# **Note**
# 
# There seems to be great approaches for collapsing variables [as seen in this post](https://stats.stackexchange.com/questions/146907/principled-way-of-collapsing-categorical-variables-with-many-levels). For simplicity I will stick with the "replace values with frequencies less than x" approach.

# In[81]:


company_df = eda_df.groupby("company", as_index = False).agg({"client_id": "count"})[["company", "client_id"]].rename(columns = {"client_id": "rows_count"})
company_df.sort_values("rows_count", ascending = False).head()


# In[68]:


company_df.describe()


# We can observe that at least 75% of the 49178 listed companies have a single row (client_id) in our dataset, and the company with most rows is Smith PLC (97 rows).
# 

# In[79]:


# companies_x_employees = company_df["rows_count"].value_counts()
# companies_x_employees = companies_x_employees.reset_index().rename(columns = {"index": "num_employees", "rows_count": "num_companies"})
more_20_emp_companies = company_df[company_df["rows_count"] > 20]
print("By grouping all companies with 20 or more employees we are now left with: " + str(len(more_20_emp_companies.company.unique())) + " companies")


# Let's replace the values from the company column of the companies with less than 20 employees with the string "Not Relevant".

# In[70]:


companies_to_replace = more_20_emp_companies.company.unique()
eda_df.loc[eda_df.company.isin(companies_to_replace) == False, "company"] = "Not Relevant"
data["train"]["merged_train"].loc[eda_df.company.isin(companies_to_replace) == False, "company"] = "Not Relevant"
print("Number of unique values in company column: \n" + str(len(eda_df.company.unique())))

# These are the 60 company names plus the "Not Relevant" value.

# ### Exploring *current_job* column
# 
# We can analyze the number of times a job appears in our training dataset to group the more relevant values (values for wich we may identify a clear pattern).

# In[82]:


job_df = eda_df.groupby("current_job", as_index = False).agg({"client_id": "count"})[["current_job", "client_id"]].rename(columns = {"client_id": "rows_count"})
job_df.sort_values("rows_count", ascending = False).head()


# In[83]:


job_df.describe()

# We can notice that the job with less rows has 81 rows, meaning that leaving all the 639 jobs as they are might be viable.


# ### Exploring *credit_card_provider* column
# 
# *credit_card_provider* column seems to be a candidate for one-hot encoding without preprocessing, as it would generate 10 columns.
# 
# Let's explore the number of times each value appears in the trainset.

# In[77]:


ccp_df = eda_df.groupby("credit_card_provider", as_index = False).agg({"client_id": "count"})[["credit_card_provider", "client_id"]].rename(columns = {"client_id": "rows_count"})
ccp_df.sort_values("rows_count", ascending = False)


# In[74]:

ccp_df.describe()

# As the credit card provider with less rows is 5714 and the one with more rows has 11572, the data is not imbalanced and we can leave the column values as they are.

# ## Exploring numerical data

# Let's analyze numerical column values.

# In[74]:

print("We have " +  str(len(num_cols)) + " numerical columns in our dataset:")
for col in num_cols:
    print("* " + col)

eda_df[num_cols].describe()

# ## Plotting pearson correlation matrix between features

# Some numerical columns might have multicollinearity, so let's plot a heatmap of the pearson correlation coefficient (PCC) between features to identify them.

# In[74]:

# Calculate the correlation matrix
corr = data["train"]["merged_train"].corr()

# plot the heatmap
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            linewidths=.5, annot = False,
            cmap="RdBu_r", center = 0)

# The columns with PCC greater than 0.3 are:

# In[74]:

for i in range(len(corr)):
    for col in corr.columns:
        if (corr.loc[corr.index[i], col] >= 0.6) & (corr.index[i] != col):
            print("Column " + col + " and column " + str(corr.index[i]) + " have a PCC of: " + str(corr.loc[corr.index[i], col]))

# Plotting pairplots

# In[74]:

for col in num_cols[0:2]:
    print(col)
    g = sns.FacetGrid(eda_df, hue="defaulted_loan")
    g.map(sns.distplot, col, kde = True)
    g.add_legend()

g = sns.PairGrid(eda_df[num_cols[0:3]+["defaulted_loan"]], hue = "defaulted_loan")
g.map_diag(sns.distplot)
g.map_offdiag(plt.scatter, s = 2)
g.add_legend()


# # Feature extraction after Exploratory Data Analysis (EDA)
# 
# A well designed EDA may lead us to the creation of relevant features.
# 
# The goal of this section is to example the implementation of two features derived from EDA:
# * salary_per_children
# * salary_per_education_year

# # Building and comparing models performance
# 
# ## Defining train and dev sets
# 
# It is necessary to train the model my measuring it's performance on not seen data (usually called dev set).
# We will assign 80% of the original train set data to our newly defined train set and the rest of 20% data to the dev set.

# In[ ]:
# Shuffle train set


x, y = shuffle(data["train"]["merged_train"].drop(["defaulted_loan"], axis = 1), data["train"]["merged_train"]["defaulted_loan"])
data["train"]["merged_train"].drop("client_id")
data["train"]["merged_train"] =data["train"]["merged_train"].drop(["client_id"], axis = 1)
x_train, x_dev, y_train, y_dev = train_test_split(x,
                                                  y,
                                                  test_size = 0.2)
categoric_trans = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), cat_cols)],
                                   remainder = "passthrough", sparse_threshold = 0)

ohe = categoric_trans.fit(x_train.append(x_dev))
x_train = ohe.transform(x_train)
x_dev = ohe.transform(x_dev)

sm = SMOTE()
x_train, y_train = sm.fit_sample(x_train, y_train)
# Assign 80% data to train set 20% data to dev set

# We can notice that our dataset's label to predict has imbalanced data because only a small fraction of observations are actually positives (the same is true if only a small fraction of observations were negatives).Recently, oversampling the minority class observations has become a common approach to improve the quality of predictive modeling. By oversampling, models are sometimes better able to learn patterns that differentiate classes. [2]

# In[ ]:

# ## Defining data wrangling pipeline steps

# ### 1) Null values Imputing
# 
# Theo need to do imputing as the trainset and testset do not contain missing values 
# 
# ### 2) Encoding categorical columns

# In[ ]:


# ## Fitting pipelines to the dataset

# ### 1) Logistic Regression Model pipeline

# In[ ]:

parameters = {'C': [1.0, 1.1]}

CV = GridSearchCV( LogisticRegression(), param_grid = parameters, scoring = 'roc_auc', n_jobs= 1)
CV.fit(x_train, y_train)

print('Best score and parameter combination = ')
print(CV.best_score_)
print(CV.best_params_)

y_pred = CV.predict(x_test)
print('MAE on validation set: %s' % (round(MAE(y_test, y_pred), 5)))

gd_sr = GridSearchCV(estimator = lr_pip,
                     param_grid=parameters,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)


# ### 2) Random Forest Model

# In[ ]:

parameters = {'': []}

CV = GridSearchCV(RandomForestClassifier(), parameters, scoring = 'roc_auc', n_jobs= 1)
CV.fit(x_train, y_train)   

print('Best score and parameter combination = ')
print(CV.best_score_)    
print(CV.best_params_)

y_pred = CV.predict(x_test)
print('MAE on validation set: %s' % (round(MAE(y_test, y_pred), 5)))

gd_sr = GridSearchCV(estimator = rf_pipe,
                     param_grid=parameters,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

# ## Making predictions and comparing the models performance

# Choosing a model of the goals of implementation. In this situation, the tradeoff between identifying more delinquent loans at the cost of misclassification can be analyzed with a specific tool called a roc curve.  When the model predicts a class label, a probability threshold is used to make the decision. This threshold is set by default at 50% so that observations with more than a 50% chance of membership belong to one class and vice-versa.

# Roc curves allow us to see the impact of varying this voting threshold by plotting the true positive prediction rate against the false positive prediction rate for each threshold value between 0% and 100%.

# The area under the ROC curve (AUC) quantifies the model’s ability to distinguish between delinquent and non-delinquent observations.  A completely useless model will have an AUC of .5 as the probability for each event is equal. A perfect model will have an AUC of 1 as it is able to perfectly predict each class.

# # Error analysis
# 
# In this section we further explore the instances in which the model made a wrong prediction to try to find patters, generate model improval propositions and measure the time investment / reward ratio of each to take a decision of the next step to perform.

# # ELI5

import eli5
eli5.show_weights(lr_pipe["lr"])
eli5.show_weights(rf_pipe["rf"])
eli5.show_prediction(lr_pipe["lr"])
eli5.show_prediction(rf_pipe["rf"])


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
# [2] https://beckernick.github.io/oversampling-modeling/
# 
# [3] https://stats.stackexchange.com/questions/146907/principled-way-of-collapsing-categorical-variables-with-many-levels
# 
# [4] https://riskspan.com/news-insight-blog/hands-on-machine-learning-predicting-loan-delinquency/