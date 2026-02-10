import pandas as pd

train_file = 'training.csv'   
test_file = 'test.csv'  

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print(df_train.head(5))
print(df_train.tail(5))

