#!/usr/bin/env python
# coding: utf-8

# # Task-1

# # Question 1: Car Matrix Generation

# In[39]:


import pandas as pd 
import numpy as np


# In[40]:


df=pd.read_csv("C:/Users/Kunal/Downloads/MapUp-Data-Assessment-F-main/datasets\dataset-1.csv")


# In[41]:


df.head()


# In[42]:


def generate_car_matrix(data):
    
    # Pivot the DataFrame to create the car matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    
    # Set diagonal values to 0
    np.fill_diagonal(car_matrix.values, 0)
    
    return pd.DataFrame(car_matrix)

result_matrix = generate_car_matrix(df)


# In[43]:


# show results 
result_matrix.head(10)


# # Question 2: Car Type Count Calculation

# In[67]:


def get_type_count(data):
    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                              labels=['low', 'medium', 'high'])
    
    # Calculate the count of occurrences for each car_type category
    type_count = df['car_type'].value_counts().sort_index().to_dict()
    
    return type_count

# Example usage
type_count_result = get_type_count(df)
print(type_count_result)


# In[68]:


df.head()


# # Question 3: Bus Count Index Retrieval

# In[47]:


def get_bus_indexes(df):
    # Identify indices where bus values are greater than twice the mean
    mean_bus_value = 2 * df['bus'].mean()
    bus_indexes = df[df['bus'] > mean_bus_value].index.tolist()
    
    return sorted(bus_indexes)

bus_indexes_result = get_bus_indexes(df)
print(bus_indexes_result)


# # Question 4: Route Filtering

# In[49]:


def filter_routes(df):
    # Filter routes based on the average of truck column
    filtered_routes = df.groupby('route')['truck'].mean()
    filtered_routes = filtered_routes[filtered_routes > 7].index.tolist()
    
    return sorted(filtered_routes)


filtered_routes_result = filter_routes(df)
print(filtered_routes_result)


# # Question 5: Matrix Value Modification

# In[51]:


def multiply_matrix(matrix):
    # Modify values based on the specified logic
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    
    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)
    
    return modified_matrix

modified_matrix_result = multiply_matrix(result_matrix)
print(modified_matrix_result.head())


# # Question 6: Time Check

# In[53]:


import pandas as pd

def time_check(df2):
    # Convert timestamp columns to datetime, handling errors and NaT (Not a Time) values
    df2['start_timestamp'] = pd.to_datetime(df2['startDay'] + ' ' + df2['startTime'], errors='coerce')
    df2['end_timestamp'] = pd.to_datetime(df2['endDay'] + ' ' + df2['endTime'], errors='coerce')
    
    # Check if timestamps cover a full 24-hour period and span all 7 days
    def check_time_interval(group):
        time_diff = group['end_timestamp'].max() - group['start_timestamp'].min()
        correct_time_interval = time_diff == pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
        return pd.Series({'correct_time_interval': correct_time_interval})
    
    result = df2.groupby(['id', 'id_2']).apply(check_time_interval)
    
    return result


df2 = pd.read_csv('C:/Users/Kunal/Downloads/MapUp-Data-Assessment-F-main/datasets\dataset-2.csv')
time_check_result = time_check(df2)
print(time_check_result)


