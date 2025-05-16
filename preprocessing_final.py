import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
import random
state = random.getstate()
random.setstate(state)

accident = pd.read_csv("accident.csv")
vehicle = pd.read_csv("vehicle.csv")
person = pd.read_csv("person.csv")
atmospheric_cond = pd.read_csv("atmospheric_cond.csv")
road_surface_cond = pd.read_csv("road_surface_cond.csv")


'''
CUSTOM METHOD
'''

def proportional_imputation(df, column):
    valid_values = df[df[column] != -1][column]
    proportions = valid_values.value_counts(normalize=True)

    missing_count = (df[column] == -1).sum()

    replacements = np.random.choice(proportions.index, missing_count, p=proportions.values).astype(int)
    
    df.loc[df[column] == -1, column] = replacements

def proportional_imputation_string(df, column):
    valid_values = df[df[column].notna()][column]
    proportions = valid_values.value_counts(normalize=True)

    missing_count = df[column].isna().sum()

    replacements = np.random.choice(proportions.index, missing_count, p=proportions.values)

    df.loc[df[column].isna(), column] = replacements

'''
ACCIDENT
'''
# ACCIDENT_NO
accident_df = pd.DataFrame({'ACCIDENT_NO': accident['ACCIDENT_NO']})

'''PERSON'''
person_df = pd.DataFrame({'ACCIDENT_NO': accident['ACCIDENT_NO']})

# SEATING POSITION
def seating_position_transform(pos):
    if pos in ['D', 'CF', 'LF', 'PL', 'CR', 'LR', 'PS', 'RR', 'OR']:
        return pos
    else:
        return np.nan

person_df['SEATING_POSITION'] = person['SEATING_POSITION'].apply(lambda x: seating_position_transform(x))

seating_position = person_df.groupby('ACCIDENT_NO')['SEATING_POSITION'].apply(lambda x: x.sample(1).iloc[0])
accident_df['SEATING_POSITION'] = accident_df['ACCIDENT_NO'].map(seating_position)
proportional_imputation_string(accident_df, 'SEATING_POSITION')

# HELMET_BELT_WORN
def helmet_belt_worn_transform(worn):
    if worn in [1, 3, 6]: # worn protection
        return 1
    else:
        return 0
    
person_df['HELMET_BELT_WORN'] = person['HELMET_BELT_WORN'].apply(lambda x: helmet_belt_worn_transform(x))
# 1: worn, 0: not worn (assuming all cases)
sum_helmet_belt_worn = person_df.groupby('ACCIDENT_NO')['HELMET_BELT_WORN'].sum()
accident_df['HELMET_BELT_WORN'] = accident_df['ACCIDENT_NO'].map(sum_helmet_belt_worn)
proportional_imputation(accident_df, 'HELMET_BELT_WORN')

# LICENCE_STATE

def licence_state_transform(state):
    if state in ['A', 'B', 'D', 'N', 'O', 'Q', 'S', 'T', 'V', 'W']:
        return state
    else:
        return 'Z'
    
person_df['LICENCE_STATE'] = person['LICENCE_STATE'].apply(lambda x: licence_state_transform(x))
# if a person involved in the accident is a driver
licence_state = person_df.groupby('ACCIDENT_NO')['LICENCE_STATE'].apply(lambda x: 1 if 1 in x.values else 0)
accident_df['LICENCE_STATE'] = accident_df['ACCIDENT_NO'].map(licence_state)
proportional_imputation_string(accident_df, 'LICENCE_STATE')

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

# SURFACE_COND
def surface_cond_transform(cond):
    if cond < 9:
        return cond
    else:
        return -1

mode_surface_cond = road_surface_cond.groupby('ACCIDENT_NO')['SURFACE_COND'].agg(lambda x: x.mode().iloc[0])
accident_df['SURFACE_COND'] = accident_df['ACCIDENT_NO'].map(mode_surface_cond)
proportional_imputation(accident_df, 'SURFACE_COND')

# ATMOSPH_COND
def atmosph_cond_transform(cond):
    if cond < 9:
        return cond
    else:
        return -1

mode_atmosph_cond = atmospheric_cond.groupby('ACCIDENT_NO')['ATMOSPH_COND'].agg(lambda x: x.mode().iloc[0])
accident_df['ATMOSPH_COND'] = accident_df['ACCIDENT_NO'].map(mode_atmosph_cond)
proportional_imputation(accident_df, 'ATMOSPH_COND')

# SEVERITY
def categorize_severity(severity):
    if severity == 1:
        return 'FATAL'
    elif severity == 2:
        return 'SERIOUS'
    else:
        return 'OTHER'

accident_df['SEVERITY'] = accident['SEVERITY'].apply(lambda x: categorize_severity(x))

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
proportional_imputation(accident_df, 'SPEED_ZONE')

# ROAD_GEOMETRY
def road_geometry_transform(road_geometry):
    if road_geometry < 9:
        return road_geometry
    else:
        return -1
    
accident_df['ROAD_GEOMETRY'] = accident['ROAD_GEOMETRY'].apply(lambda x: road_geometry_transform(x))
proportional_imputation(accident_df, 'ROAD_GEOMETRY')

# NO_PERSONS_NOT_INJ
def persons_not_inj_check(num):
    if num > -1:
        return num
    else:
        return -1
    
accident_df['NO_PERSONS_NOT_INJ'] = accident['NO_PERSONS_NOT_INJ'].apply(lambda x: persons_not_inj_check(x))
proportional_imputation(accident_df, 'NO_PERSONS_NOT_INJ')

# TAKEN_HOSPITAL
def taken_hospital(code):
    if pd.isna(code):
        return np.nan
    else:
        return code

person_df['TAKEN_HOSPITAL'] = person['TAKEN_HOSPITAL'].apply(lambda x: taken_hospital(x))

taken_hospital_cond= person_df.groupby('ACCIDENT_NO')['TAKEN_HOSPITAL'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
accident_df['TAKEN_HOSPITAL'] = accident_df['ACCIDENT_NO'].map(taken_hospital_cond)
proportional_imputation_string(accident_df, 'TAKEN_HOSPITAL')

# AGE_GROUP
def age_group_transform(age_group):
    if age_group == "0-4" or age_group == "5-12" or age_group == "13-15":
        return "Under 16"
    elif age_group == "16-17" or age_group == "18-21" or age_group == "22-25":
        return "17-25"
    elif age_group == "26-29" or age_group == "30-39":
        return "26-39"
    elif age_group == "40-49" or age_group == "50-59" or age_group == "60-64":
        return "40-64"
    elif age_group == "65-69" or age_group == "70+":
        return "65+"
    else:
        return np.nan

person_df['AGE_GROUP'] = person['AGE_GROUP'].apply(lambda x: age_group_transform(x))
proportional_imputation_string(person_df, 'AGE_GROUP')

age_order = ["Under 16", "17-25", "26-39", "40-64", "65+", "Unknown"]

person_df['AGE_GROUP_ORDERED'] = person_df['AGE_GROUP'].map(lambda x: age_order.index(x))

def median_age_group(ages):
    median_index = int(np.median(sorted(ages)))
    return age_order[median_index]

age_group_median = (
    person_df.groupby('ACCIDENT_NO')['AGE_GROUP_ORDERED']
    .apply(lambda x: median_age_group(x))
    .reset_index(name='MEDIAN_AGE_GROUP')
)

accident_df = accident_df.merge(age_group_median, on='ACCIDENT_NO', how='right')

# DAY_OF_WEEK
def day_of_week_code(day):
    if 1 <= day <= 7:
        return day
    else:
        return -1
    
accident['DAY_OF_WEEK'] = accident['DAY_OF_WEEK'].apply(lambda x: day_of_week_code(x))
accident_df['DAY_OF_WEEK'] = accident_df['ACCIDENT_NO'].map(accident.set_index('ACCIDENT_NO')['DAY_OF_WEEK'])
proportional_imputation(accident_df, 'DAY_OF_WEEK')

# NO_OF_VEHICLES
def no_of_vehicles(num):
    if num >= 0:
        return num
    else:
        return -1

accident['NO_OF_VEHICLES'] = accident['NO_OF_VEHICLES'].apply(lambda x: no_of_vehicles(x))
accident_df['NO_OF_VEHICLES'] = accident_df['ACCIDENT_NO'].map(accident.set_index('ACCIDENT_NO')['NO_OF_VEHICLES'])
proportional_imputation(accident_df, 'NO_OF_VEHICLES')

# NO_PERSONS

def no_persons_code(code):
    if code > -1:
        return code
    else:
        return -1

accident['NO_PERSONS'] = accident['NO_PERSONS'].apply(lambda x: no_persons_code(x))
accident_df['NO_PERSONS'] = accident_df['ACCIDENT_NO'].map(accident.set_index('ACCIDENT_NO')['NO_PERSONS'])
proportional_imputation(accident_df, 'NO_PERSONS')

accident_df_copy = accident_df.copy()

# aggregate after imputation
def agg_light_surf_atmos(light_cond, surface_cond, atmosph_cond):
    return (light_cond + surface_cond + atmosph_cond) / 3

accident_df['AGG_LIGH_SURF_ATMOS_COND'] = accident_df[['LIGHT_CONDITION', 'SURFACE_COND', 'ATMOSPH_COND']].mean(axis=1).round(2)

# Remove LIGHT_CONDITION, SURFACE_COND, ATMOSPH_COND because we aggregated them
accident_df = accident_df.drop(columns=['LIGHT_CONDITION', 'SURFACE_COND', 'ATMOSPH_COND']) 

'''
drop columns we don't need anymore: because correlation is negligible
'''

'''
VEHICLE
'''

vehicle_df = pd.DataFrame({'ACCIDENT_NO': vehicle['ACCIDENT_NO']})

# VEHICLE_YEAR_MANUF
def vehicle_year_manuf_transform(year):
    if pd.isna(year):
        return -1
    else:
        return year
    
vehicle_df['VEHICLE_YEAR_MANUF'] = vehicle['VEHICLE_YEAR_MANUF'].apply(lambda x: vehicle_year_manuf_transform(x)).astype(int)

proportional_imputation(vehicle_df, 'VEHICLE_YEAR_MANUF')
mode_year = vehicle_df.groupby('ACCIDENT_NO')['VEHICLE_YEAR_MANUF'].agg(lambda x: x.mode().iloc[0])
accident_df['VEHICLE_YEAR_MANUF'] = accident_df['ACCIDENT_NO'].map(mode_year)

# FUEL_TYPE
def fuel_type_transform(fuel_type):
    if fuel_type == 'Z':
        return np.nan
    else:
        return fuel_type
    
vehicle_df['FUEL_TYPE'] = vehicle['FUEL_TYPE'].apply(lambda x: fuel_type_transform(x))

proportional_imputation_string(vehicle_df, 'FUEL_TYPE')
mode_fuel_type = vehicle_df.groupby('ACCIDENT_NO')['FUEL_TYPE'].agg(lambda x: x.mode().iloc[0])
accident_df['VEHICLE_FUEL_TYPE'] = accident_df['ACCIDENT_NO'].map(mode_fuel_type)

# TARE_WEIGHT
def tare_weight_transform(tare_weight):
    if pd.isna(tare_weight):
        return -1
    else:
        return tare_weight
    
vehicle_df['TARE_WEIGHT'] = vehicle['TARE_WEIGHT'].apply(lambda x: tare_weight_transform(x))

proportional_imputation(vehicle_df, 'TARE_WEIGHT')
mean_tare_weight = vehicle_df.groupby('ACCIDENT_NO')['TARE_WEIGHT'].mean().astype(int)
accident_df['VEHICLES_TARE_WEIGHT'] = accident_df['ACCIDENT_NO'].map(mean_tare_weight)

# ROAD_SURFACE_TYPE
def road_surface_type_transform(road_surface_type):
    if road_surface_type == 9:
        return -1
    else:
        return road_surface_type
    
vehicle_df['ROAD_SURFACE_TYPE'] = vehicle['ROAD_SURFACE_TYPE'].apply(lambda x: road_surface_type_transform(x))

proportional_imputation(vehicle_df, 'ROAD_SURFACE_TYPE')
mode_road_surface_type = vehicle_df.groupby('ACCIDENT_NO')['ROAD_SURFACE_TYPE'].agg(lambda x: x.mode().iloc[0])
accident_df['AGG_ROAD_SURFACE_TYPE'] = accident_df['ACCIDENT_NO'].map(mode_road_surface_type)

# CAUGHT_FIRE
def caught_fire_transform(caught_fire):
    if caught_fire == 9 or caught_fire == 0:
        return -1
    else:
        return caught_fire
    
vehicle_df['CAUGHT_FIRE'] = vehicle['CAUGHT_FIRE'].apply(lambda x: caught_fire_transform(x))

proportional_imputation(vehicle_df, 'CAUGHT_FIRE')
caught_fire_ans = vehicle_df.groupby('ACCIDENT_NO')['CAUGHT_FIRE'].apply(lambda x: 1 if 1 in x.values else 2) # if one value is 1, all values will be 1; similar for 2
accident_df['CAUGHT_FIRE']  = accident_df['ACCIDENT_NO'].map(caught_fire_ans)

# LEVEL OF DAMAGE
def level_of_damage_transform(level):
    if level == 6:
        return 0
    elif level == 9:
        return -1
    else:
        return level
    
vehicle_df['LEVEL_OF_DAMAGE'] = vehicle['LEVEL_OF_DAMAGE'].apply(lambda x: level_of_damage_transform(x))

proportional_imputation(vehicle_df, 'LEVEL_OF_DAMAGE')
mode_level_of_damage = vehicle_df.groupby('ACCIDENT_NO')['LEVEL_OF_DAMAGE'].agg(lambda x: x.mode().iloc[0])
accident_df['HIGHEST_LEVEL_OF_DAMAGE'] = accident_df['ACCIDENT_NO'].map(mode_level_of_damage)
accident_df['HIGHEST_LEVEL_OF_DAMAGE'] = accident_df['HIGHEST_LEVEL_OF_DAMAGE'].fillna(lambda x: np.random.randint(1, 6))

# List of columns to convert
columns_to_convert = ['VEHICLE_YEAR_MANUF', 'AGG_ROAD_SURFACE_TYPE', 'HIGHEST_LEVEL_OF_DAMAGE', 'CAUGHT_FIRE']

accident_df.dropna(how='any', inplace=True)

accident_df[columns_to_convert] = accident_df[columns_to_convert].astype(int)
accident_df.drop('LICENCE_STATE', axis=1, inplace=True)

# categorical data
# accident_df.to_csv("accident_processed_new(20).csv", index=False)

'''
numerical data
'''
accident_numerical = pd.read_csv("accident_processed_new(20).csv")

accident_df_numerical = accident_df.copy()

def numerical_seating_position(pos):
    if pos == 'D':
        return 1
    elif pos in ['CF', 'LF', 'PL', 'CR', 'LR', 'PS', 'RR', 'OR']:
        return 0
    else:
        return -1
    
accident_df_numerical['SEATING_POSITION'] = accident_df_numerical['SEATING_POSITION'].apply(lambda x: numerical_seating_position(x))
proportional_imputation(accident_df_numerical, 'SEATING_POSITION')

def numerical_taken_hospital(taken):
    if taken == 'Y':
        return 1
    elif taken == 'N':
        return 0
    else: 
        return -1
    
accident_df_numerical['TAKEN_HOSPITAL'] = accident_df_numerical['TAKEN_HOSPITAL'].apply(lambda x: numerical_taken_hospital(x))
proportional_imputation(accident_df_numerical, 'TAKEN_HOSPITAL')

def numerical_fuel_type(fuel):
    if fuel == 'D':
        return 1
    elif fuel == 'E':
        return 2
    elif fuel == 'G':
        return 3
    elif fuel == 'M':
        return 4
    elif fuel == 'P':
        return 5
    elif fuel == 'R':
        return 6
    else:
        return -1

accident_df_numerical['VEHICLE_FUEL_TYPE'] = accident_df_numerical['VEHICLE_FUEL_TYPE'].apply(lambda x: numerical_fuel_type(x))
proportional_imputation(accident_df_numerical, 'VEHICLE_FUEL_TYPE')

accident_df_numerical.dropna(how='any', inplace=True)

accident_df["VEHICLES_TARE_WEIGHT_BINNED"] = pd.cut(accident_df["VEHICLES_TARE_WEIGHT"], bins=[0, 1000, 3000, 5000, np.inf], labels=["0-1000", "1001-3000", "3001-5000", "5000-7000"])

accident_df.dropna(how='any', inplace=True)

# print(accident_df['VEHICLES_TARE_WEIGHT_BINNED'].head(30))

# replace the 0 values with the integer mean of all other non-zero entries
accident_df['VEHICLE_YEAR_MANUF'] = accident_df['VEHICLE_YEAR_MANUF'].replace(0, int(accident_df.loc[accident_df['VEHICLE_YEAR_MANUF'] != 0, 'VEHICLE_YEAR_MANUF'].mean()))
accident_df_numerical['VEHICLE_YEAR_MANUF'] = accident_df_numerical['VEHICLE_YEAR_MANUF'].replace(0, int(accident_df_numerical.loc[accident_df_numerical['VEHICLE_YEAR_MANUF'] != 0, 'VEHICLE_YEAR_MANUF'].mean()))

# print(accident_df['VEHICLE_YEAR_MANUF'].min())
# print(accident_df['VEHICLE_YEAR_MANUF'].max())

accident_df["VEHICLE_YEAR_MANUF_BINNED"] = pd.cut(accident_df["VEHICLE_YEAR_MANUF"], bins=[1900, 1980, 2005, 2024, np.inf], labels=["1900-1980", "1981-2005", "2006-2024", "2024-2025"])

# save

# accident_df.to_csv("for_correlation_analysis.csv", index=False)

accident_df.to_csv("for_supervised_learning_model_categorical.csv", index=False)
# accident_df_numerical.to_csv("for_supervised_learning_model_numerical.csv", index=False)
