import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import acquire
from env import username, password, host

############################ Outliers #############################

def remove_outliers(df, k, col_list):
    ''' this function take in a dataframe, k value, and specified columns 
    within a dataframe and then return the dataframe with outliers removed
    '''
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

############################# Clean ################################

def clean_zillow(df):
    '''
    This function takes in the zillow data, cleans it, and returns a dataframe
    '''
    
    # Apply a function to remove outliers
    df = remove_outliers(df, 2.7, ['bedrooms','bathrooms',
                                   'sqft', 'lot_sqft','tax_value'])
    
    # Remove further outliers for sqft to ensure data is usable
    df = df[(df.sqft > 500) & (df.sqft < 3_000)]
    # Remove further outliers for taxvalue to ensure data is usable
    df = df[(df.tax_value > 1_000) & (df.tax_value < 1_000_000)]
    # Remove further outliners with no bathrooms
    df = df[(df.bathrooms >1)]
    
    #Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()

    # Create list of datatypes to change
    int_cols = ['bedrooms','sqft','tax_value', 'yr_built', 'lot_sqft', 'fips']
  
    
    # Change data types of above columns
    for col in df:
        if col in int_cols:
            df[col] = df[col].astype(int)
#         if col in obj_cols:
#             df[col] = df[col].astype(int).astype(object)
    
    # Found the counties that Zach deleted... smh
    df.fips = df.fips.replace({6037:'Los Angeles',
                           6059:'Orange',          
                           6111:'Ventura'})
    # Rename 'fips' to 'county
    df.rename(columns={'fips':'county'}, inplace = True)
    
    # Drop the target column
    # df = df.drop(columns='tax_value')
    
    return df

############################# Split #################################

def split_data(df):
    ''' 
    This function will take in the data and split it into train, 
    validate, and test datasets for modeling, evaluating, and testing
    '''
    train_val, test = train_test_split(df, train_size = .8, random_state = 123)

    train, validate = train_test_split(train_val, train_size = .7, random_state = 123)

    return train, validate, test

############################# Scale ################################


def scale_data(train, validate, test):
    '''
    Scales the 3 data splits. using the MinMaxScaler()
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    If return_scaler is true, the scaler object will be returned as well.
    '''
    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit scaler on train dataset
    scaler.fit(train)

    # Transform and rename columns for all three datasets
    train_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns = train.columns.tolist())
    test_scaled = pd.DataFrame(scaler.transform(test), columns = train.columns.tolist())

    return train_scaled, validate_scaled, test_scaled

############################# Wrangle ################################

def wrangle_zillow(df):
    ''' 
    This function combines both functions above and outputs three 
    cleaned and prepped datasets
    '''
    clean_df = clean_zillow(df)
    train, validate, test = split_data(clean_df)

    return train, validate, test

######################################################################

def define_x_y(train, validate, test, train_scaled, validate_scaled, test_scaled):
    # create X and y version of train, validate, and test
    x_train = train_scaled.drop(columns= ['lot_sqft','taxvalue','yr_built','county'])
    y_train = pd.DataFrame(train.taxvalue)

    x_validate = validate_scaled.drop(columns= ['lot_sqft','taxvalue','yr_built','county'])
    y_validate = pd.DataFrame(validate.taxvalue)

    x_test = test_scaled.drop(columns= ['lot_sqft','taxvalue','yr_built','county'])
    y_test = pd.DataFrame(test.taxvalue)

    return x_train, y_train, x_validate, y_validate, x_test, y_test

def baseline_measures():
    # Turn y_train and y_validate into dataframes so we can append new columns
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    #Calculate tax_value mean
    tax_value_pred_mean = y_train['tax_value'].mean()
    y_train['tax_value_pred_mean'] = tax_value_pred_mean
    y_validate['tax_value_pred_mean'] = tax_value_pred_mean
    
    # Calculate tax_value_median
    tax_value_pred_median = y_train['tax_value'].median()
    y_train['tax_value_pred_median'] = tax_value_pred_median
    y_validate['tax_value_pred_median'] = tax_value_pred_median
    
    # Calcualte RMSE of tax_value_pred_mean
    basemean_rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean)**(1/2)
    basemean_rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean)**(1/2)
    
    print('====================================')
    print('          Baseline Measures')
    print('====================================')
    print("RMSE using Mean\nTrain/In-Sample: ", round(basemean_rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(basemean_rmse_validate, 2))
    print('------------------------------------')
    
    # Calculate RMSE of tax_value_pred_median
    basemedian_rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_median)**(1/2)
    basemedian_rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_median)**(1/2)
    
    print("RMSE using Median\nTrain/In-Sample: ", round(basemedian_rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(basemedian_rmse_validate, 2))
    print('------------------------------------')
    r2_baseline = r2_score(y_validate.tax_value, y_validate.tax_value_pred_mean)
    print(f'The r^2 score for baseline is {round(r2_baseline, 6)}')
    print('------------------------------------')