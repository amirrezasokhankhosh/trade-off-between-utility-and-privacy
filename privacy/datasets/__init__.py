
from utils.types import Dataset

def get_dataset_params(name):
    if name == Dataset.ADULT:
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        target_var = 'salary-class'
        IS_CAT = [False, True, True, True, False, True, False, True, True, True, False, False, False, True]
        max_numeric = {
            "age": 90,
            "fnlwgt" :  1484705,
            "education-num" : 16,
            "capital-gain" : 99999,
            "capital-loss" : 4356,
            "hours-per-week" : 99
        }
    elif name == Dataset.TEXAS:
        QI_INDEX = [i for i in range(1, 13)]
        target_var = "TOTAL_CHARGES"
        IS_CAT = [True for _ in range(11)] + [False]
        max_numeric = {
            "LENGTH_OF_STAY" : 1380
        }
    elif name == Dataset.OCCUPANCY:
        QI_INDEX = [i for i in range(1, 17)]
        target_var = "Room_Occupancy_Count"
        IS_CAT = [False for _ in range(16)]
        max_numeric = {
            "S1_Temp" : 2638,
            "S2_Temp" : 2900,
            "S3_Temp" : 2619,
            "S4_Temp" : 2656,
            "S1_Light" : 165,
            "S2_Light" : 258,
            "S3_Light" : 280,
            "S4_Light" : 74,
            "S1_Sound" : 388,
            "S2_Sound" : 344,
            "S3_Sound" : 367,
            "S4_Sound" : 340,
            "S5_CO2" : 1270,
            "S5_CO2_Slope" : 8980769,
            "S6_PIR" : 1,
            "S7_PIR" : 1
        }
    else:
        print(f"Not support {name} dataset")
        raise ValueError
    return {
        'qi_index': QI_INDEX,
        'is_category': IS_CAT,
        'target_var': target_var,
        'max_numeric': max_numeric
    }