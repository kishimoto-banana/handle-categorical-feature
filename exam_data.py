import pandas as pd

# カラム名の定義
ground_truth_column = ['is_click']
integer_columns = ['integer_' + str(i+1) for i in range(13)]
categorical_columns = ['categorical_' + str(i+1) for i in range(26)]

# データ読み込み
df = pd.read_csv('data/dac_sample.txt', sep='\t', header=None)
df.columns = ground_truth_column + integer_columns + categorical_columns

# 正例の割合
test_rate = 0.2
test_splited_index = int(len(df) * test_rate)
print(f'データ全体の正例の割合 = {df["is_click"].sum() / len(df)}')

df_subsample = df[-test_splited_index:]
print(f'データ全体の正例の割合 = {df_subsample["is_click"].sum() / len(df_subsample)}')

# データ全体の正例の割合 = 0.22663
# データ全体の正例の割合 = 0.23525