import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error


def trainEDA(train_merged, test_merged):
    # parse the date data
    train_merged[['year', 'month', 'day']] = train_merged['date'].astype(str).str.split("-", expand=True)
    train_merged['dayofweek'] = train_merged['date'].dt.dayofweek

    test_merged[['year', 'month', 'day']] = test_merged['date'].astype(str).str.split("-", expand=True)
    test_merged['dayofweek'] = test_merged['date'].dt.dayofweek

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

    train_merged = pd.concat([train_merged, pd.get_dummies(train_merged['family'])], axis=1)
    test_merged = pd.concat([test_merged, pd.get_dummies(test_merged['family'])], axis=1)

    train_merged['sales'].plot()
    plt.show()
    showing average sales price over the years
    train_merged_g = train_merged.reset_index()
    train_merged_g = train_merged_g.set_index('date')
    y_train_merged = train_merged_g['sales'].resample('MS').mean()
    y_train_merged.plot(figsize=(15,6))
    plt.show()

    plot unique values per column
    unique_values = train_merged.select_dtypes(include="number").nunique().sort_values()
    unique_values.plot.bar(logy=True, figsize=(15, 4), title="Unique values per feature")
    plt.show()

    plot missing values
    plt.figure(figsize=(10, 8))
    plt.imshow(train_merged.isna(), aspect="auto", interpolation="nearest", cmap="gray")
    plt.xlabel("Column Number")
    plt.ylabel("Sample Number")
    plt.show()

    count the family columns
    train_merged['family'].value_counts().plot(kind='bar')
    plt.show()

    sublots
    train_merged.plot(lw=0, marker=".", subplots=True, layout=(-1, 4), figsize=(15, 5), markersize=1)
    plt.show()

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

    enc_family = OrdinalEncoder()
    train_merged['family'] = enc_family.fit_transform(train_merged[['family']])

    dict3 = {i : train_merged[train_merged['family'] == i]['sales'].mean()  for i in train_merged['family'].unique()}

    sns.barplot(x=list(dict3.keys()), y=list(dict3.values()), palette="Blues_d").set(title='Average sales per family',
                                                                                   xlabel='Family',
                                                                                   ylabel='Average sales')
    plt.show()

    sns.barplot(data=train_merged, x = 'dcoilwtico', y = 'sales')
    plt.show()

    train_merged.pop('date')
    train_merged.pop('family')
    train_merged.pop('type')

    test_merged.pop('date')
    test_merged.pop('family')
    test_merged.pop('type')

    train_merged[['day', 'month', 'year']] = train_merged[['day', 'month', 'year']].astype(int)
    test_merged[['day', 'month', 'year']] = test_merged[['day', 'month', 'year']].astype(int)

    return train_merged, test_merged


def load_raw_data(raw_df):
    y = raw_df['sales']
    x = raw_df.drop('sales', axis=1)
    x_pca = CheckPCA(data=x , y = y)
    #x_pca = 0

    return x, x_pca, y


def CheckPCA(data, y , fisi=(10, 8)):

    pca = PCA()  # Default n_components = min(n_samples, n_features)
    pca.fit_transform(data)
    exp_var_pca = pca.explained_variance_ratio_
    Cumulative sum of eigenvalues; This will be used to create step plot
    for visualizing the variance explained by each principal component.
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    Create the visualization plot
    plt.figure(figsize=fisi)
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title(f'PCA ', fontsize=18)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    sum = 0
    c = 0
    for i in exp_var_pca:
        if sum <= 0.98:
            sum += i
            c += 1

    if (c > int(0.8 * len(exp_var_pca))):
        print("PCA not recommended")

    else:
        pca1 = PCA(n_components=c)  # contains 90% of the variance
        data = pca1.fit_transform(data)
        plot = plt.scatter(data[:, 0], y)
        plt.show()
        plt.legend(handles=plot.legend_elements()[0], labels=list(winedata['target_names']))
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(data[:, 0] , data[:, 1], y)
        threeD_view1 = ax.view_init(0,0)
        threeD_view2 = ax.view_init(45, 0)
        threeD_view3 = ax.view_init(45, 215)
        plt.show()
        plt.savefig('threeD_view1.png')
        threeD_view2.savefig('threeD_view2.png')
        threeD_view3.savefig('threeD_view3.png')
        print("PCA with ", c, "components (90% variance)")
    return data


def FindBestParams(model, space, x_train, y_train):
    res = []  # Return the saved arrays
    models = ["DecisionTreeRegressor", "RandomForestRegressor", "XGBRegressor"]

    for m in range(0, len(model)):
        print("---- Starting Gridsearch on model " , model[m])
        clf = GridSearchCV(model[m], space[m], scoring='neg_mean_squared_log_error', verbose=2)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set: ", models[m])
        print()
        print(clf.best_params_)
        print()

        # res.append([clf, models[m], model[m], clf.best_estimator_, clf.best_params_])
        res.append(clf)
    return res


def SplitPcaScale(x, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)

    return X_train, X_test, y_train, y_test


# rmsle calculation
def rmsle(y_test, predictions):
    predictions = [0 if i < 0 else i for i in predictions]
    return np.sqrt(mean_squared_log_error(y_test, predictions))


# test merged -> x train, y train
def RMSLEGraph(res, x_test, y_test):

    rmsle_train = []
    rmsle_val = []

    for key in range(0, len(res)):
        rmsle_train.append(np.sqrt(-1*res[key].best_score_))
        y_hat = res[key].best_estimator_.predict(x_test)
        rmsle_val.append(rmsle(y_test, y_hat))

    x = ["DecisionTreeRegressor", "RandomForestRegressor"]

    x_axis = np.arange(len(x))
    plt.bar(x_axis - 0.2,rmsle_val, 0.4 , label= 'RMSLE validation results')
    plt.bar(x_axis + 0.2 ,rmsle_train, 0.4, label= 'RMSLE train results')
    plt.xticks(x_axis,x)
    plt.xlabel("Used Models")
    plt.ylabel("RMSLE validaiton/train results")
    plt.title("RMSLE result comparison between models")
    plt.show()

if __name__ == '__main__':

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
    train_merged = pd.merge(train, transactions, on=['date', 'store_nbr'])
    train_merged = pd.merge(train_merged, oil, on='date')
    train_merged = pd.merge(train_merged, stores, on='store_nbr')

    test['holi'] = test['date'].isin(holiday['date']).astype(int)
    test_merged = pd.merge(test, transactions, on=['date', 'store_nbr'])
    test_merged = pd.merge(test_merged, oil, on='date')
    test_merged = pd.merge(test_merged, stores, on='store_nbr')

    train_merged = train_merged.drop(columns=['Unnamed: 0_y', 'Unnamed: 0_x'])
    test_merged = test_merged.drop(columns=['Unnamed: 0_y', 'Unnamed: 0_x'])

    train_merged, test_merged = trainEDA(train_merged, test_merged)

    raw_df = train_merged

    # define the models
    models = [DecisionTreeRegressor(), RandomForestRegressor()]

    # define the search space
    space = [
             [{"criterion": ['squared_error'], "max_depth": [30], "min_samples_leaf": [2],
               "max_features": ['auto', 'sqrt', 'log2']}],
             [{"max_depth": [30], "criterion": ['squared_error'],
               "n_estimators": [20]}]]

    x, x_pca, y = load_raw_data(raw_df)

    x_train, x_test, y_train, y_test = SplitPcaScale(x, y)
    x_pca_train, x_pca_test, y_train_pca, y_test_pca = SplitPcaScale(x_pca, y)

    res = FindBestParams(models, space, x_train, y_train)

    res_pca = FindBestParams(models, space, x_pca_train, y_train_pca)
    pca = PCA(2)  # Default n_components = min(n_samples, n_features)
    x_test_pca = pca.fit_transform(x_test)

    RMSLEGraph(res,x_test,y_test)
    # RMSLEGraph(res_pca, x_test_pca, y_test)
