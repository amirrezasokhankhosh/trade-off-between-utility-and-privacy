import os
from preprocessing import scale_encode_data
from datasets import get_dataset_params
import pandas as pd


def encodeData(data, method, k):
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(app_path, f'data/{data}')
    result_path = os.path.join(app_path, 'mondrian_results')

    data_params = get_dataset_params(data)
    QI_INDEX = data_params['qi_index']
    IS_CAT = data_params['is_category']
    ori_csv = os.path.join(data_path, f'train.csv')
    tmp_df = pd.read_csv(ori_csv, delimiter=';')
    
    pr_data, X_org, y_org = scale_encode_data(data, ori_csv, QI_INDEX, IS_CAT)
    anon_csv = os.path.join(result_path, f'{data}_anonymized_{k}.csv')
    pr_data_anon, X_anon, y_anon = scale_encode_data(
        data,
        ori_csv,
        anon_csv=anon_csv,
        qi_index=QI_INDEX, 
        is_cat=IS_CAT,
        att_trees=None)
    
    return pr_data, X_org, y_org, pr_data_anon, X_anon, y_anon

def encodeNonAnonData(data, path):
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(app_path, f'data/{data}')
    ori_csv = os.path.join(data_path, f'train.csv')
    data_params = get_dataset_params(data)
    QI_INDEX = data_params['qi_index']
    IS_CAT = data_params['is_category']
    _, _, _ = scale_encode_data(data, ori_csv, QI_INDEX, IS_CAT)
    pr_data, X, y = scale_encode_data(
        data,
        ori_csv,
        anon_csv=path,
        qi_index=QI_INDEX, 
        is_cat=IS_CAT,
        att_trees=None)
    return pr_data, X, y

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    synPah = os.path.join(path, 'data/adult_test.csv')
    
    # _, X_org, y_org, _, X_anon, y_anon = encodeData("adult", "classic_mondrian", "5")
    # X_test, y_test = encodeNonAnonData("adult", test_path)
    # print(X_test.shape)
    

