import pandas as pd
from tqdm import tqdm


def one_hot_encoding(df, qi_index, is_qi_cat):
    rows, columns = df.shape
    attributes = list(df.columns)
    is_cat_list = []
    # Get list of categorical attribute
    for idx, row in df.iterrows():
        for j, value in enumerate(row):
            if j in qi_index:
                ## If QID
                pos = qi_index.index(j)
                is_cat = is_qi_cat[pos]
            else:
                try:
                    float(value)
                    is_cat = False
                except ValueError:
                    # Is categorical attribute
                    is_cat = True
            is_cat_list.append(is_cat)
        break

    # Name of categorical attributes
    cat_attrs = [attributes[idx] for idx,i in enumerate(is_cat_list) if i]

    # One hot encoding all categorical attribtutes
    df = pd.get_dummies(df, columns=cat_attrs)
    
    return df

def embed_target(targets):
    unique_labels = set(targets)
    label_to_idx = {v:i for i, v in enumerate(unique_labels)} 
    new_targets = [label_to_idx[i] for i in targets]
    return new_targets, label_to_idx

def replace_generalization(anon_df, columns, qi_index=None, is_cat=None, att_trees=None):
    """
    Replace all generalized value to its mean
    """

    def get_non_qid_value(key, value):
        try:
            return float(value), 0
        except:
            return key+'_'+value, 1

    def get_mean(value):
        if isinstance(value, str):
            tmp = value.split('~')
            if len(tmp) == 2:
                low, high = tmp
                mean = float(low) + (float(high) - float(low))/2
                return mean
            else:
                if '*' in value:
                    low = float(value.replace('*', '0'))
                    high = float(value.replace('*', '9'))
                    return low+(high-low)/2
                elif not value.isnumeric():
                    return 0
        return float(value)

    def get_caterogical_value(key, value, att_trees=None):
        if att_trees is None:
            value = str(value)
            value_splits = value.split('~')
        else:
            value_splits = att_trees[str(value)].get_leaves_names()

        return [key+'_'+i for i in value_splits]

    tmp_list = []
    for _, row in tqdm(anon_df.iterrows()):
        atr_dict = {
            key:0 for key in columns
        }
        for atr_idx, key in enumerate(list(row.keys())):
            value = row[atr_idx]
            # If not QID, append value
            if atr_idx not in qi_index:
                new_key, is_category = get_non_qid_value(key, value)
                if is_category:
                    atr_dict[new_key] = 1
                else:
                    atr_dict[key] = new_key
                continue
            else:
                # If is QID
                # If is categorical
                qi_id = qi_index.index(atr_idx)
                if is_cat[qi_id]:
                    tmp = att_trees[qi_id] if att_trees is not None else None
                    keys = get_caterogical_value(key, value, tmp)
                    for new_key in keys:
                        atr_dict[new_key] = 1
                else:
                    # If is numeric
                    mean = get_mean(value)
                    atr_dict[key] = mean
        tmp_list.append(atr_dict)

    result_dict = {
        k:[] for k in columns
    }

    for item in tmp_list:
        for atr in result_dict.keys():
            result_dict[atr].append(item[atr])
    
    new_df = pd.DataFrame(result_dict)
    return new_df