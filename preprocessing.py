import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
import random

accident = pd.read_csv("accident.csv")
# vehicle = pd.read_csv("vehicle.csv", nrows=10000)
person = pd.read_csv("person.csv")
# filtered_vehicle = pd.read_csv("filtered_vehicle_subset.csv")
atmospheric_cond = pd.read_csv("atmospheric_cond.csv")
road_surface_cond = pd.read_csv("road_surface_cond.csv")

# dataset for pre-processing

'''
1. Add UNIQUE values of ACCIDENT_NO to ACCIDENT_ID and modify the entire dataset accordingly
'''

'''
CHANGE OF PLAN: INDEPENDENTLY FORM DATAFRAME WITH PERSON DATASET AND DATAFRAME WITH ACCIDENT DATASET, and then figure out what to do next
'''

'''PERSON'''
# person_df = person[['ACCIDENT_NO']].drop_duplicates().reset_index(drop=True)
person_df = pd.DataFrame({'ACCIDENT_NO': accident['ACCIDENT_NO']})

# SEATING POSITION
# change acc to seating score
def seating_position_transform(pos):
    if pos == 'D':
        return 2
    elif pos in ['CF', 'LF', 'PL', 'CR', 'LR', 'PS', 'RR', 'OR']:
        return 1
    else:
        return -1
    
person_df['SEATING_POSITION'] = person['SEATING_POSITION'].apply(lambda x: seating_position_transform(x))

# HELMET_BELT_WORN

def helmet_belt_worn_transform(worn):
    if worn in [1, 3, 6]: # worn protection
        return 1
    else:
        return 0
    
person_df['HELMET_BELT_WORN'] = person['HELMET_BELT_WORN'].apply(lambda x: helmet_belt_worn_transform(x))

# LICENCE_STATE

def licence_state_transform(state):
    if state != 'Z' or state != '':
        return 1
    else:
        return 0
    
person_df['LICENCE_STATE'] = person['LICENCE_STATE'].apply(lambda x: licence_state_transform(x))


'''
ACCIDENT
'''
# ACCIDENT_NO
accident_df = pd.DataFrame({'ACCIDENT_NO': accident['ACCIDENT_NO']})

# LIGHT_CONDITION
def light_condition_transform(light):
    if light == 9:
        return 0
    elif light == 6:
        return 4
    elif light == 4:
        return 5
    else:
        return light

accident_df['LIGHT_CONDITION'] = accident['LIGHT_CONDITION'].apply(lambda x: light_condition_transform(x))

# SEVERITY

accident_df['SEVERITY'] = accident['SEVERITY']


# SPEED_ZONE

def speed_zone_transform(zone):
    if zone == 777:
        return random.choice([10,20])
    elif zone == 888:
        return 10
    elif zone == 999:
        return -1
    else:
        return zone
    
accident_df['SPEED_ZONE'] = accident['SPEED_ZONE'].apply(lambda x: speed_zone_transform(x))

# ROAD_GEOMETRY

def road_geometry_transform(road_geometry):
    if road_geometry == 9:
        return -1
    else:
        return road_geometry
    
accident_df['ROAD_GEOMETRY'] = accident['ROAD_GEOMETRY'].apply(lambda x: road_geometry_transform(x))

# NO_PERSONS_NOT_INJ

accident_df['NO_PERSONS_NOT_INJ'] = accident['NO_PERSONS_NOT_INJ']

# combine person_df to accident_df

accident_df[['LICENCE_STATE', 'SEATING_POSITION', 'HELMET_BELT_WORN', 'SURFACE_COND', 'ATMOSPH_COND']] = None

# fill the columns up

# LICENCE_STATE (see line 50)
'''if a person involved in the accident is a driver'''
licence_state = person_df.groupby('ACCIDENT_NO')['LICENCE_STATE'].apply(lambda x: 1 if (x > 0).any() else 0)
accident_df['LICENCE_STATE'] = accident_df['ACCIDENT_NO'].map(licence_state)

# SEATING_POSITION (see line 26)
'''
Seating position score:
Front passenger: 2 (more weight to avoid skew)
Rear Passenger: 1 (no. of passengers >= driver)
Answer types:
+ve: High severity
~0: Balanced
-ve: Low Severity
Finding overall severity of accident, not individual person injury severity
Given that all entries have resulted in an accident
'''

seating_position_score = person_df.groupby('ACCIDENT_NO')['SEATING_POSITION'].sum()
accident_df['SEATING_POSITION'] = accident_df['ACCIDENT_NO'].map(seating_position_score)

# HELMET_BELT_WORN (see line 38)
'''
Define a function that counts how many persons did not wear helment/belt
More people who did not wear helmet/belt, higher the injury severity
Lower the number of people wearing helmet/seatbelt, lower the injury severity
'''
# 1: worn, 0: not worn (assuming all cases)

avg_helmet_belt_worn = person_df.groupby('ACCIDENT_NO')['HELMET_BELT_WORN'].sum()
accident_df['HELMET_BELT_WORN'] = accident_df['ACCIDENT_NO'].map(avg_helmet_belt_worn)

# SURFACE_COND
mode_surface_cond = road_surface_cond.groupby('ACCIDENT_NO')['SURFACE_COND'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
accident_df['SURFACE_COND'] = accident_df['ACCIDENT_NO'].map(mode_surface_cond)

# missing values and unknown surface_cond
accident_df['SURFACE_COND'].replace(9, 0, inplace=True) 
accident_df['SURFACE_COND'].fillna(-1, inplace=True)

# ATMOSPH_COND
mode_atmosph_cond = atmospheric_cond.groupby('ACCIDENT_NO')['ATMOSPH_COND'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
accident_df['ATMOSPH_COND'] = accident_df['ACCIDENT_NO'].map(mode_atmosph_cond)

# missing values and unknown atmosph_cond
accident_df['ATMOSPH_COND'].replace(9, 0, inplace=True) 
accident_df['ATMOSPH_COND'].fillna(-1, inplace=True)


'''
Fill in missing values in such a way that the values filled in maintain
the original proportion of non-missing values.
20% 'kill' and 80% 'not kill' for non-missing values:
20% 'kill' and 80% 'not kill' for all values after imputation
'''

def proportional_imputation(df, column):
    valid_values = df[df[column] != -1][column]
    proportions = valid_values.value_counts(normalize=True)

    missing_count = (df[column] == -1).sum()

    replacements = np.random.choice(proportions.index, missing_count, p=proportions.values)
    
    df.loc[df[column] == -1, column] = replacements

# apply the custom method
proportional_imputation(accident_df, 'SPEED_ZONE')
proportional_imputation(accident_df, 'ROAD_GEOMETRY')
proportional_imputation(accident_df, 'SEATING_POSITION')
proportional_imputation(accident_df, 'HELMET_BELT_WORN')


# ADD SURFACE_COND AND ATMOSPH_COND TO A NEW DATAFRAME

surface_atmosph_df = accident_df[['SURFACE_COND', 'ATMOSPH_COND']].copy()
accident_df.drop(columns=['SURFACE_COND', 'ATMOSPH_COND'], inplace=True)

proportional_imputation(surface_atmosph_df, 'SURFACE_COND')
proportional_imputation(surface_atmosph_df, 'ATMOSPH_COND')

accident_df_copy = accident_df.copy()



norm_cols = ['LIGHT_CONDITION', 'SEVERITY', 'SPEED_ZONE', 'ROAD_GEOMETRY', 'NO_PERSONS_NOT_INJ', 'SEATING_POSITION']


normalised_accident_df = pd.DataFrame(
    MinMaxScaler().fit_transform(accident_df_copy[norm_cols]),
    columns=norm_cols
)

normalised_accident_df.insert(0, 'ACCIDENT_NO', accident_df['ACCIDENT_NO'].values)

'''
normalised_accident_df['HELMET_BELT_WORN'] = accident_df['HELMET_BELT_WORN']
normalised_accident_df['LICENCE_STATE'] = accident_df['LICENCE_STATE']
'''

# NO_OF_VEHICLES
accident_df['NO_OF_VEHICLES'] = accident['NO_OF_VEHICLES']

# normalised_accident_df['NO_OF_VEHICLES'] = accident_df['NO_OF_VEHICLES']

# TAKEN_HOSPITAL
# normalised_accident_df['TAKEN_HOSPITAL'] = accident_df['TAKEN_HOSPITAL'] 

# print(normalised_accident_df.head(50))

# print(accident_df.head(50))
accident_df.to_csv("accident_processed.csv", index=False)

'''
add:
1. NO_OF_VEHICLES

'''