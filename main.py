import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv('train.csv')
print(train.info())
oil = pd.read_csv('oil_final.csv')
stores = pd.read_csv('stores_final.csv')
transactions = pd.read_csv('transactionsF.csv')
test = pd.read_csv('test.csv')
holiday = pd.read_csv('holidays_final.csv')

holiday['date'] = pd.to_datetime(holiday['date'])
train['date'] = pd.to_datetime(train['date'])
transactions['date'] = pd.to_datetime(transactions['date'])
oil['date'] = pd.to_datetime(oil['date'])
test['date'] = pd.to_datetime(test['date'])



train['holi'] = train['date'].isin(holiday['date']).astype(int)
train_merged = pd.merge(train,transactions,on=['date','store_nbr'])
train_merged = pd.merge(train_merged,oil,on='date')
train_merged = pd.merge(train_merged,stores,on='store_nbr')

test['holi'] = test['date'].isin(holiday['date']).astype(int)
test_merged = pd.merge(test,transactions,on=['date','store_nbr'])
test_merged = pd.merge(test_merged,oil,on='date')
test_merged = pd.merge(test_merged,stores,on='store_nbr')

train_merged=train_merged.drop(columns=['Unnamed: 0_y','Unnamed: 0_x'])
test_merged=test_merged.drop(columns=['Unnamed: 0_y','Unnamed: 0_x'])



# train_merged.to_csv("train_merged.csv")
# test_merged.to_csv("test_merged.csv")

def trainEDA(train_merged):

    #parse the date data
    train_merged[['year', 'month', 'day']] = train_merged['date'].astype(str).str.split("-", expand=True)
    train_merged['dayofweek'] = train_merged['date'].dt.dayofweek

    train_merged_c = train_merged.copy()
    train_merged_c.pop('A')
    train_merged_c.pop('B')
    train_merged_c.pop('C')
    train_merged_c.pop('D')
    train_merged_c.pop('E')
    train_merged_c.pop('id')
    train_merged_c.pop('date')
    train_merged_c.pop('city')
    train_merged_c.pop('cluster')
    train_merged_c.pop('state')
    train_merged_c.pop('family')
    train_merged_c.pop('gradient_3')
    train_merged_c.pop('gradient_4')
    train_merged_c.pop('gradient_5')
    train_merged_c.pop('gradient_6')
    train_merged_c.pop('gradient_7')
    train_merged_c.pop('gradient_8')
    train_merged_c.pop('gradient_9')
    train_merged_c.pop('gradient_10')


    df_corr = train_merged_c.corr(method="pearson")

    labels = np.where(np.abs(df_corr) > 0.75, "S",
                      np.where(np.abs(df_corr) > 0.5, "M", np.where(np.abs(df_corr) > 0.25, "W", " ")))

    plt.figure(figsize=(10, 10))
    sns.heatmap(df_corr, mask=np.eye(len(df_corr)), square=True, center=0, annot=labels, fmt='', linewidths=0.5,
                cmap="vlag", cbar_kws={"shrink": 0.8})
    plt.show()

    # creating dummies from the family column
    train_merged = pd.concat([train_merged, pd.get_dummies(test_merged['family'])] ,axis=1)



    print(train_merged.info())

    train_merged['sales'].plot()
    plt.show()
    # showing average sales price over the years
    train_merged_g = train_merged.reset_index()
    train_merged_g = train_merged_g.set_index('date')
    y_train_merged = train_merged_g['sales'].resample('MS').mean()
    y_train_merged.plot(figsize=(15,6))
    plt.show()

    #plot unique values per column
    unique_values = train_merged.select_dtypes(include="number").nunique().sort_values()
    unique_values.plot.bar(logy=True, figsize=(15, 4), title="Unique values per feature")
    plt.show()

    #plot missing values
    plt.figure(figsize=(10, 8))
    plt.imshow(train_merged.isna(), aspect="auto", interpolation="nearest", cmap="gray")
    plt.xlabel("Column Number")
    plt.ylabel("Sample Number")
    plt.show()

    #count the family columns
    train_merged['family'].value_counts().plot(kind='bar')
    plt.show()

    #sublots
    # train_merged.plot(lw=0, marker=".", subplots=True, layout=(-1, 4), figsize=(15, 5), markersize=1)
    # plt.show()


    dict1 = {i : train_merged[train_merged['city'] == i]['sales'].mean()  for i in train_merged['city'].unique()}

    sns.barplot(x=list(dict1.keys()), y=list(dict1.values()), palette="Blues_d").set(title='Average sales per city',
                                                                                   xlabel='City',
                                                                                   ylabel='Average sales')
    plt.show()

    dict2 = {i : train_merged[train_merged['store_nbr'] == i]['sales'].mean()  for i in train_merged['store_nbr'].unique()}

    sns.barplot(x=list(dict2.keys()), y=list(dict2.values()), palette="Blues_d").set(title='Average sales per store',
                                                                                   xlabel='Store number',
                                                                                   ylabel='Average sales')
    plt.show()


    # enc_family = OrdinalEncoder()
    # train_merged['family'] = enc_family.fit_transform(train_merged[['family']])

    dict3 = {i : train_merged[train_merged['family'] == i]['sales'].mean()  for i in train_merged['family'].unique()}

    sns.barplot(x=list(dict3.keys()), y=list(dict3.values()), palette="Blues_d").set(title='Average sales per family',
                                                                                   xlabel='Family',
                                                                                   ylabel='Average sales')
    plt.show()




    # sns.barplot(data=train_merged, x = 'dcoilwtico', y = 'sales')
    # plt.show()

    train_merged.pop('date')
    train_merged.pop('family')
    train_merged.pop('type')

    train_merged[['day', 'month', 'year']] =  train_merged[['day', 'month', 'year']].astype(int)
    print(train_merged.info())




trainEDA(train_merged)




