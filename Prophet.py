# Get time series data
import yfinance as yf
# Prophet model for time series forecast
from prophet import Prophet
# Data processing
import numpy as np
import pandas as pd
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Model performance evaluation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from main_create_data import trainEDA
import warnings
warnings.filterwarnings("ignore")


def rmsle(y_test, predictions):
    return np.sqrt(mean_squared_log_error(y_test, predictions))

def main():

    start_date = '2016-01-01'
    test_end_date = '2017-08-31'
    train_end_date = '2017-08-01'
    validation_end_date = '2017-08-15'

    ## read csv data.
    train = pd.read_csv('train.csv')
    # print(train.info())
    oil = pd.read_csv('oil_final.csv')
    stores = pd.read_csv('stores_final.csv')
    transactions = pd.read_csv('transactionsF.csv')
    test_csv = pd.read_csv('test.csv')
    holiday = pd.read_csv('holidays_final.csv')

    ## change date columns to date format.
    holiday['date'] = pd.to_datetime(holiday['date'], infer_datetime_format=True)
    train['date'] = pd.to_datetime(train['date'], infer_datetime_format=True)
    transactions['date'] = pd.to_datetime(transactions['date'], infer_datetime_format=True)
    oil['date'] = pd.to_datetime(oil['date'], infer_datetime_format=True)
    test_csv['date'] = pd.to_datetime(test_csv['date'], infer_datetime_format=True)

    ## mereing the csv data to train test
    train['holi'] = train['date'].isin(holiday['date']).astype(int)
    train_merged = pd.merge(train, transactions, on=['date', 'store_nbr'])
    train_merged = pd.merge(train_merged, oil, on='date')
    train_merged = pd.merge(train_merged, stores, on='store_nbr')

    test_csv['holi'] = test_csv['date'].isin(holiday['date']).astype(int)
    test_merged = pd.merge(test_csv, transactions, on=['date', 'store_nbr'])
    test_merged = pd.merge(test_merged, oil, on='date')
    test_merged = pd.merge(test_merged, stores, on='store_nbr')

    train_merged = train_merged.drop(columns=['Unnamed: 0_y', 'Unnamed: 0_x'])
    test_merged = test_merged.drop(columns=['Unnamed: 0_y', 'Unnamed: 0_x'])

    df = train_merged
    df2 = test_merged

    df.rename(columns={'holi': 'holiday', 'sales': 'y', 'date': 'ds'}, inplace=True)
    scores_df = pd.DataFrame(columns=['store', 'category', 'score', 'date', 'id'])
    ddf = pd.DataFrame(columns=['true', 'preds'])

    submission = pd.read_csv('sample_submission.csv')
    submit = []
    test_csv['sales'] = -1

    for i in df['store_nbr'].unique():
        for category in df[df['store_nbr'] == i]['family'].unique():

            d_f = df[(df['store_nbr'] == i) & (df['family'] == category)]
            # df = df[['y', 'ds','id', 'store_nbr','onpromotion', 'holiday', 'transactions',
            #        'dcoilwtico', 'gradient_3', 'gradient_4', 'gradient_5', 'gradient_6',
            #        'gradient_7', 'gradient_8', 'gradient_9', 'gradient_10', 'city',
            #        'state', 'cluster', 'Populaiton', 'A', 'B', 'C', 'D', 'E', 'year',
            #        'month', 'day', 'dayofweek', 'AUTOMOTIVE', 'BABY CARE', 'BEAUTY',
            #        'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CELEBRATION', 'CLEANING',
            #        'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II',
            #        'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II',
            #        'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN',
            #        'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
            #        'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS',
            #        'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD']]
            list_of_predictors = ['dcoilwtico', 'ds', 'onpromotion', 'transactions', 'gradient_3',
                       'gradient_4', 'gradient_5', 'gradient_6', 'gradient_7', 'gradient_8',
                       'gradient_9', 'gradient_10', 'city', 'state', 'cluster', 'Populaiton',
                       'A', 'B', 'C', 'D', 'E']

            # data = df[list_of_predictors]
            d_f['ds'] = pd.to_datetime(d_f['ds'])

            # Train test split
            train = d_f[(d_f['ds'] <= train_end_date) & (d_f['ds'] >= start_date)]
            validation = d_f[((d_f['ds'] <= validation_end_date) & (d_f['ds'] > train_end_date))]
            #test = d_f[d_f['ds'] > end_date]

            # Check the shape of the dataset
            print(train.shape)
            print(validation.shape)
            # print(test.shape)

            holidays = d_f[d_f['holiday'] == 1]
            holidays['lower_window'] = 0
            holidays['upper_window'] = 1
            holidays['holiday'] = 'holiiiishit'
            model = Prophet(holidays=holidays[['holiday', 'ds']])
            for col in list_of_predictors:
                if col == 'y' or col == 'ds':
                    continue
                model.add_regressor(col)

            model.fit(train)

    #         forecast_validation = model.predict(validation.drop(columns="y"))
    # #        forecast_test = model.predict(test.drop(columns = "y"))
    #         # model.plot(forecast)
    #         # plt.show()
    #         # print(forecast)
            future_baseline = model.make_future_dataframe(periods=30)
            future_baseline = pd.merge(left=future_baseline, right=d_f, how='left', on='ds')
            future_baseline = future_baseline.interpolate(method='ffill')
            future_baseline['dcoilwtico'] = future_baseline['dcoilwtico'].fillna(future_baseline['dcoilwtico'].mean())
            # Make prediction
            forecast_baseline = model.predict(future_baseline)

            #Visualize the forecast
            model.plot(forecast_baseline)
            plt.show()

            val_len = len(validation)
            #preds_val = forecast_baseline[((forecast_baseline['ds'] <= validation_end_date) & (forecast_baseline['ds'] >= train_end_date))]
            preds_val = forecast_baseline[forecast_baseline['ds'].isin(validation['ds'])]
            #preds_val = forecast_baseline.iloc[len(forecast_baseline)-31:len(forecast_baseline)-16]

            preds_test = forecast_baseline[((forecast_baseline['ds'] <= test_end_date) & (forecast_baseline['ds'] > validation_end_date))]
            #preds_test = forecast_baseline.iloc[len(forecast_baseline)-16:]


            for index, row in preds_test.iterrows():
                # test_csv.iloc[]
                # test_csv.loc[(test_csv['date'].isin([row[0]])) & (test_csv['store_nbr'].isin([i]))
                #              & (test_csv['family'].isin([category]))].iloc[0]['sales'] = row[-1]

                test_csv.at[test_csv.index[(test_csv['date'] == row[0]) & (test_csv['store_nbr'] == i) & (
                            test_csv['family'] == category)][0], 'sales'] = row[-1]
                # test_csv.loc[(test_csv['date'].isin([row[0]])) & (test_csv['store_nbr'].isin([i])) & (test_csv['family'].isin([category]))]['sales'][0] = row[-1]

            for sub in preds_test['yhat']:
                submit.append(sub)

            s = pd.Series()
            for index, val in enumerate(preds_val['yhat']):
                if val < 0:
                    s = s.append(pd.Series([0]))
                else:
                    s = s.append(pd.Series([val]))

            x = rmsle(validation['y'], s)
            print("rmsle=", x, " for store ", i, " in ", category, " category ")
            scores_df = scores_df.append({'category': category, 'score': x, 'store': i, }, ignore_index=True)

            for true, pred in zip(validation['y'], s):
                ddf = ddf.append({'true': true, 'preds': pred}, ignore_index=True)

    test_csv.to_csv('b4_submission.csv')
    print(scores_df['score'].mean())
    x = rmsle(ddf['true'], ddf['preds'])
    print("total rmsle = ", x)
    #submission['sales'] = submit
    #submission.to_csv('submissionProphet.csv')

if __name__ =="__main__":
    main()



#---- add a graph of the each store and the mean RMSLE !









# df = df.drop(df.columns[:4], axis=1)

#
# def to_date(row):
#     if len(str(int(row.month))) == 1 and len(str(int(row.day))) == 1:
#         return str(int(row.year)) + "-0" + str(int(row.month)) + "-0" + str(int(row.day))
#     elif len(str(int(row.month))) == 2 and len(str(int(row.day))) == 1:
#         return str(int(row.year)) + "-" + str(int(row.month)) + "-0" + str(int(row.day))
#     elif len(str(int(row.month))) == 1 and len(str(int(row.day))) == 2:
#         return str(int(row.year)) + "-0" + str(int(row.month)) + "-" + str(int(row.day))
#     elif len(str(int(row.month))) == 2 and len(str(int(row.day))) == 2:
#         return str(int(row.year)) + "-" + str(int(row.month)) + "-" + str(int(row.day))
#
#
# df['date'] = df.apply(lambda x: to_date(x), axis=1)
# df['date'] = pd.to_datetime(df['date'])
# df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

# df.to_csv("new_data.csv")

# sns.lineplot(x=df.ds, y=df['y'])
# plt.show()

#
# list_of_categories = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY',
#        'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CELEBRATION', 'CLEANING',
#        'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II',
#        'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II',
#        'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN',
#        'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
#        'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS',
#        'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD']

#
