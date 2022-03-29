
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from env import username, password, host

def acquire_zillow(use_cache=True):
    ''' 
    This function acquires all necessary housing data from zillow 
    needed to better understand future pricing
    '''
    
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/zillow'
    query = '''
    SELECT bedroomcnt AS bedrooms, 
           bathroomcnt AS bathrooms, 
           calculatedfinishedsquarefeet AS sqft, 
           taxvaluedollarcnt AS tax_value, 
           yearbuilt AS yr_built,
           taxamount AS tax_amount,
           regionidcounty AS county_id,
           fips
        FROM properties_2017
    
        JOIN propertylandusetype USING(propertylandusetypeid)
        
        JOIN predictions_2017 pr USING (parcelid)
        WHERE propertylandusedesc IN ('Single Family Residential',
        
                                      'Inferred Single Family Residential')
                              AND pr.transactiondate LIKE '2017%%';
            '''
    
    
    df = pd.read_sql(query, database_url_base)
    df.to_csv('zillow.csv', index=False)
   
    return df