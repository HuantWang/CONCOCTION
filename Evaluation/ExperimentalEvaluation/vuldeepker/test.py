
import pandas as pd
#读取工作簿和工作簿中的工作表
data_frame=pd.read_excel("result/CWE-200-1649653106_result.xlsx",sheet_name="Sheet1")
seed = data_frame['seed'][0]
print()


import pandas as pd

df = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv', parse_dates=['saledate'], low_memory=False)


from fast_ml.model_development import train_valid_test_split

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'SalePrice',
                                                                            method='sorted', sort_by_col='saledate',
                                                                            train_size=0.8, valid_size=0.1, test_size=0.1)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)