import numpy as np
import pandas as pd
from .preprocess import (one_hot_encoding, 
    replace_generalization, embed_target)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def scale_encode_data(
    data,
    raw_csv,
    qi_index, 
    is_cat,
    scalars=None, 
    att_trees=None,
    anon_csv = None): 

    df = pd.read_csv(raw_csv, delimiter=';')
    
    # Drop ID and Target columns (last column)
    df = df.drop(['ID'], axis=1)
    targets = list(df.iloc[:, -1])
    df = df.drop(df.columns[-1], axis=1)

    # Because we remove ID column
    qi_index = [i-1 for i in qi_index]

    # One-hot categorical values
    one_hot_df = one_hot_encoding(df, qi_index, is_cat)

    # print(one_hot_df.head())
    # One-hot target labels
    if data != 'texas':
        embeded_targets, label_to_idx = embed_target(targets)

    if anon_csv is not None:
        anon_df = pd.read_csv(anon_csv, delimiter=';')
        anon_df = anon_df.drop(['ID'], axis=1)
        anon_df = anon_df.drop(anon_df.columns[-1], axis=1)

        print("Replacing all generalized values...")
        one_hot_anon_df = replace_generalization(
            anon_df, 
            columns=list(one_hot_df.columns),
            qi_index=qi_index,
            is_cat=is_cat,
            att_trees=att_trees)

    if anon_csv is not None:
        union = list(one_hot_anon_df.index)
        features = one_hot_anon_df.loc[union]
        if data == "adult":
            features_save = features.copy()
            num = features[["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]].copy()
            features.drop(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"], axis=1, inplace=True)
            if not scalars:
                scalar = StandardScaler()
                num = pd.DataFrame(scalar.fit_transform(num), columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"])
            else:
                scalar = scalars
                num = pd.DataFrame(scalar.transform(num), columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"])
            features = pd.concat([num, features], axis=1)
            targets = [embeded_targets[i] for i in union]
            targets_df = pd.DataFrame(np.array(targets).T, columns=["salary_class"])
            pr_data = pd.concat([features_save, targets_df], axis=1)
        elif data == "texas":
            features_save = features.copy()
            num = features[["LENGTH_OF_STAY"]].copy()
            features.drop(["LENGTH_OF_STAY"], axis=1, inplace=True)
            if not scalars:
                scalar = StandardScaler()
                num = pd.DataFrame(scalar.fit_transform(num), columns=["LENGTH_OF_STAY"])
            else:
                scalar = scalars[0]
                num = pd.DataFrame(scalar.transform(num), columns=["LENGTH_OF_STAY"])
            features = pd.concat([num, features], axis=1)
            targets_df = pd.DataFrame(np.array(targets).T, columns=["TOTAL_CHARGES"])
            if not scalars:
                scalar_out = StandardScaler()
                targets = pd.DataFrame(scalar_out.fit_transform(np.array(targets).reshape((-1, 1))), columns=["TOTAL_CHARGES"])
            else:
                scalar_out = scalars[1]
                targets = pd.DataFrame(scalar_out.transform(np.array(targets).reshape((-1, 1))), columns=["TOTAL_CHARGES"])
            pr_data = pd.concat([features_save, targets_df], axis=1)
        else:
            times_10 = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp", "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]
            times_1000000 = ["S5_CO2_Slope"]
            for attr in times_10:
                features[attr] = features[attr].apply(lambda x: x / 100)
            for attr in times_1000000:
                features[attr] = features[attr].apply(lambda x: x / 1000000)

            features_save = features.copy()

            if not scalars:
                scalar = StandardScaler()
                features = pd.DataFrame(scalar.fit_transform(features), columns=features.columns)
            else:
                scalar = scalars
                features = pd.DataFrame(scalar.transform(features), columns=features.columns)
            
            targets = [embeded_targets[i] for i in union]
            targets_save = pd.DataFrame(targets, columns=["Room_Occupancy_Count"])
            encoder = OneHotEncoder()
            targets = encoder.fit_transform(np.array(targets).reshape(len(targets), 1)).toarray()
            pr_data = pd.concat([features_save, targets_save], axis=1)

        return None, pr_data, np.array(features), np.array(targets)
    

    else:
        union = list(one_hot_df.index)
        features = one_hot_df.loc[union]
        if data == "adult":
            features_save = features.copy()
            num = features[["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]].copy()
            features.drop(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"], axis=1, inplace=True)
            scalar = StandardScaler()
            num = pd.DataFrame(scalar.fit_transform(num), columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"])
            features = pd.concat([num, features], axis=1)
            targets = [embeded_targets[i] for i in union]
            targets_df = pd.DataFrame(np.array(targets).T, columns=["salary_class"])
            pr_data = pd.concat([features_save, targets_df], axis=1)
            scalars = scalar
        elif data == "texas":
            features_save = features.copy()
            num = features[["LENGTH_OF_STAY"]].copy()
            features.drop(["LENGTH_OF_STAY"], axis=1, inplace=True)
            scalar = StandardScaler()
            num = pd.DataFrame(scalar.fit_transform(num), columns=["LENGTH_OF_STAY"])
            features = pd.concat([num, features], axis=1)
            targets_df = pd.DataFrame(np.array(targets).T, columns=["TOTAL_CHARGES"])
            scalar_out = StandardScaler()
            targets = pd.DataFrame(scalar_out.fit_transform(np.array(targets).reshape((-1, 1))), columns=["TOTAL_CHARGES"])
            pr_data = pd.concat([features_save, targets_df], axis=1)
            scalars = [scalar, scalar_out]
        else:
            scalar = StandardScaler()

            times_10 = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp", "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]
            times_1000000 = ["S5_CO2_Slope"]
            for attr in times_10:
                features[attr] = features[attr].apply(lambda x: x / 100)
            for attr in times_1000000:
                features[attr] = features[attr].apply(lambda x: x / 1000000)

            features_save = features.copy()
            features = pd.DataFrame(scalar.fit_transform(features), columns=features.columns)
            targets = [embeded_targets[i] for i in union]
            targets_save = pd.DataFrame(targets, columns=["Room_Occupancy_Count"])
            encoder = OneHotEncoder()
            targets = encoder.fit_transform(np.array(targets).reshape(len(targets), 1)).toarray()
            
            pr_data = pd.concat([features_save, targets_save], axis=1)
            scalars = scalar
        
        return scalars, pr_data, np.array(features), np.array(targets)
