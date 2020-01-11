
# # Building and comparing models performance

# ## Defining data wrangling pipeline steps
# 
# ### 1) Null values Imputing
# 
# Theo need to do imputing as the trainset and testset do not contain missing values 
# 
# ### 2) Encoding categorical columns

# In[ ]:
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
x, y = shuffle(data["train"]["merged_train"].drop(["defaulted_loan"], axis = 1), data["train"]["merged_train"]["defaulted_loan"])
# Assign 80% data to train set 20% data to dev set
x_train, x_dev, y_train, y_dev = train_test_split(x,
                                                  y,
                                                  test_size = 0.2)



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
