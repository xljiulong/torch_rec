import functools
import pandas as pd
import numpy as np
import json

feature_json_file = '/data/snlp/zhangjl/datas/ctr/criteo_sample_50w_feat_map.json'
c_data = pd.read_csv('/data/snlp/zhangjl/datas/ctr/criteo_sample_50w.csv')
sample_file = '/data/snlp/zhangjl/datas/ctr/criteo_sample_50w_train_sample.csv'

c_data.head()

def generate_feature_map_file():
    readable_df = pd.DataFrame()
    for col in c_data.columns:
        if col != 'label':
            if str(col).startswith('I'):
                readable_df[col] = c_data[col].apply(lambda x: f'{col}:{x}' if pd.notna(x) else np.nan)
            else:
                readable_df[col] = c_data[col].apply(lambda x: f'{col}_{x}:1' if  pd.notna(x) else np.nan)


    print(readable_df.head())


    final_set = set()
    for col in readable_df.columns:
        if col != 'label':
            tmp_series = readable_df[col].apply(lambda x: str(x).split(':')[0])
            tmp_set = set(tmp_series.drop_duplicates().tolist())
            final_set = final_set.union(tmp_series)

    final_set.remove('nan')

    final_list = list(final_set)
    final_list.sort()

    feature_series = pd.Series(final_list)

    print(len(feature_series))

    feature_json = {}
    for index, value in feature_series.items():
        t_index = index + 1
        feature_json[value] = t_index

    dst_json_obj = json.dumps(feature_json)
    with open(feature_json_file, 'w') as wf:
        wf.write(dst_json_obj)

    
# generate_feature_map_file()
with open(feature_json_file, 'r') as rf:
    feature_json = json.load(rf)
print(len(feature_json))


def gen_cat_feat(x):
    x = c_data[col]
    # print(x)
    # print('...')
    for i in x:
        if pd.notna(i):
            key = f'{col}_{i}'
            return f'{feature_json[key]}:1'
        return np.nan
    
    # if pd.notna(x):
    #     key = f'{col}_{x}'
    #     return f'{feature_json[key]}:1'
    # return np.nan
    

readable_df = pd.DataFrame()
for col in c_data.columns:
    if col == 'label':
        readable_df[col] = c_data[col]
    if col != 'label':
        if str(col).startswith('I'):
            readable_df[col] = c_data[col].apply(lambda x: f'{feature_json[col]}:{x}' if pd.notna(x) else np.nan)
        else:
            # readable_df[col] = c_data[col].apply(gen_cat_feat)
            readable_df[col] = c_data[col].apply(lambda x: str(feature_json[col+'_'+x]) + ':1' if pd.notna(x) else np.nan)


print(readable_df.head())

def sort_feat(fa, fb):
    va = int(fa.split(':')[0])
    vb = int(fb.split(':')[0])

    if va > vb:
        return 1
    
    return -1
with open(sample_file, 'w', encoding='utf-8') as wf:
    for index, row in readable_df.iterrows():
        index = index + 1
        if index % 10000 == 0:
            print(f'processing line {index}')
        
        label = row['label']
        feature_lst = []
        for feature in readable_df.columns:
            if feature == 'label':
                continue
            feature_value = row[feature]
            feature_lst.append(feature_value)
        feature_lst = list(filter(lambda x: 0 if pd.isna(x) else 1, feature_lst))
        if len(feature_lst) <= 0:
            continue
        
        feature_lst.sort(key= functools.cmp_to_key(sort_feat))
        feat_str = ' '.join(feature_lst)
        sample = f'{label} {feat_str}\n'
        wf.write(sample)
        wf.flush()

