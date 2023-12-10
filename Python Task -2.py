# # Task-2

# In[54]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[55]:


df3=pd.read_csv('C:/Users/Kunal/Downloads/MapUp-Data-Assessment-F-main/datasets\dataset-3.csv')


# In[56]:


df3.head()


# In[57]:


def calculate_distance_matrix(df3):
    # Pivot the data to get a matrix-like structure
    pivoted_data = df3.pivot(index='id_start', columns='id_end', values='distance').fillna(0)
    
    # Convert to a symmetric matrix by adding its transpose
    distance_matrix = pivoted_data + pivoted_data.T
    
    # Set diagonal values to 0
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0
    
    return pd.DataFrame(distance_matrix)


distance_matrix_result = calculate_distance_matrix(df3)
print(distance_matrix_result.head())


# In[58]:


#Question 2: Unroll Distance Matrix


# In[59]:


def unroll_distance_matrix(distance_matrix):
    # Create an empty DataFrame to store unrolled distances
    unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])
    
    # Iterate through each unique combination of id_start and id_end
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Exclude entries where id_start and id_end are the same
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_distances = unrolled_distances.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance
                }, ignore_index=True)
    
    return unrolled_distances


unrolled_distances_result = unroll_distance_matrix(distance_matrix_result)
print(unrolled_distances_result)


# In[60]:


#Question 3: Finding IDs within Percentage Threshold


# In[61]:


def find_ids_within_ten_percentage_threshold(unrolled_distances, reference_value):
    # Filter rows with the specified reference value
    reference_rows = unrolled_distances[unrolled_distances['id_start'] == reference_value]
    
    # Calculate the average distance for the reference value
    reference_average = reference_rows['distance'].mean()
    
    # Calculate the threshold range (within 10%)
    lower_threshold = reference_average - (0.1 * reference_average)
    upper_threshold = reference_average + (0.1 * reference_average)
    
    # Filter rows within the threshold range and get unique values from id_start column
    within_threshold_ids = unrolled_distances[
        (unrolled_distances['distance'] >= lower_threshold) & 
        (unrolled_distances['distance'] <= upper_threshold)
    ]['id_start'].unique()
    
    # Sort the result
    within_threshold_ids = sorted(within_threshold_ids)
    
    return within_threshold_ids

# Example usage with unrolled_distances_result from Question 2 and a reference value
reference_value = 1001400  # Replace with the desired reference value
result_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distances_result, reference_value)
print(result_within_threshold)


# In[62]:


#Question 4: Calculate Toll Rate


# In[63]:


import pandas as pd

def calculate_toll_rate(unrolled_distances):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        unrolled_distances[vehicle_type] = unrolled_distances['distance'] * rate_coefficient
    
    return unrolled_distances

# Example usage with unrolled_distances_result from Question 2
result_with_toll_rate = calculate_toll_rate(unrolled_distances_result)
print(result_with_toll_rate)


# In[64]:


#Question 5: Calculate Time-Based Toll Rates


# In[65]:


import datetime

def calculate_time_based_toll_rates(unrolled_distances):
    # Convert 'id_start' and 'id_end' to datetime objects
    unrolled_distances['id_start'] = pd.to_datetime(unrolled_distances['id_start'])
    unrolled_distances['id_end'] = pd.to_datetime(unrolled_distances['id_end'])

    # Define time ranges and discount factors
    time_ranges = [
        {'id_start': datetime.time(0, 0, 0), 'id_end': datetime.time(10, 0, 0), 'discount_factor': 0.8},
        {'id_start': datetime.time(10, 0, 0), 'id_end': datetime.time(18, 0, 0), 'discount_factor': 1.2},
        {'id_start': datetime.time(18, 0, 0), 'id_end': datetime.time(23, 59, 59), 'discount_factor': 0.8}
    ]

    # Extract day names from datetime objects
    unrolled_distances['start_day'] = unrolled_distances['id_start'].dt.day_name()
    unrolled_distances['end_day'] = unrolled_distances['id_end'].dt.day_name()

    # Extract time from datetime objects
    unrolled_distances['id_start'] = unrolled_distances['id_start'].dt.time
    unrolled_distances['id_end'] = unrolled_distances['id_end'].dt.time

    # Iterate through time ranges and apply discount factors
    for time_range in time_ranges:
        mask = (unrolled_distances['id_start'] >= time_range['id_start']) & (unrolled_distances['id_end'] <= time_range['id_end'])
        unrolled_distances.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= time_range['discount_factor']

    # Apply a constant discount factor of 0.7 for weekends
    weekend_mask = (unrolled_distances['start_day'].isin(['Saturday', 'Sunday']))
    unrolled_distances.loc[weekend_mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= 0.7

    return unrolled_distances

result_with_time_based_toll_rates = calculate_time_based_toll_rates(unrolled_distances_result)
print(result_with_time_based_toll_rates)


# In[ ]:




