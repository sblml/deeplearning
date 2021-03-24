import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_data(phase_flag='3'):
    """returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """

    df = pd.read_csv('df_chapter82_processed.csv', index_col='Date')
    feature1_list = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'log_return']
    feature2_list = ['CCI', 'MA5', 'MA10', 'RASD5', 'RASD10', 'MACD', 'ATR', 'ub', 'lb', 'MTM1', 'MTM3', 'ROC', 'WPR']
    feature3_list = ['S&P500', 'SOX', 'VIX']
    feature4_list = ['next_rtn']
    all_features = feature1_list + feature2_list + feature3_list + feature4_list
    # phase_flag = '3'

    if phase_flag == '1' :
        train_from = '2010-01-04'
        train_to = '2012-01-01'

        val_from = '2012-01-01'
        val_to = '2012-04-01'

        test_from = '2012-04-01'
        test_to = '2012-07-01'

    elif phase_flag == '2' :
        train_from = '2012-07-01'
        train_to = '2014-07-01'

        val_from = '2014-07-01'
        val_to = '2014-10-01'

        test_from = '2014-10-01'
        test_to = '2015-01-01'
        
    else : 
        train_from = '2015-01-01'
        train_to = '2017-01-01'

        val_from = '2017-01-01'
        val_to = '2017-04-01'

        test_from = '2017-04-01'
        test_to = '2017-07-01'

    train_df = df.loc[train_from: train_to, :].copy()
    val_df = df.loc[val_from: val_to, :].copy()
    test_df = df.loc[test_from: test_to, :].copy()

    train_sample_df, eng_list =  min_max_normal(train_df, all_features, feature4_list)
    val_sample_df, eng_list =  min_max_normal(val_df, all_features, feature4_list)
    test_sample_df, eng_list = min_max_normal(test_df, all_features, feature4_list)

    X_train, y_train = create_dataset(train_sample_df, eng_list, target='next_rtn', steps=5)
    X_val, y_val = create_dataset(val_sample_df, eng_list, target='next_rtn', steps=5)
    X_test, y_test = create_dataset(test_sample_df, eng_list, target='next_rtn', steps=5)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def min_max_normal(tmp_df, all_features, feature4_list):
    eng_list = []
    sample_df = tmp_df.copy()
    for x in all_features:
        if x in feature4_list:
            continue
        values = sample_df[x].copy().values
        values = values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler = scaler.fit(values)
        normalized = scaler.transform(values)
        new_feature = f"{x}_normal"
        eng_list.append(new_feature)
        sample_df[new_feature] = normalized
    return sample_df, eng_list

def create_dataset(data, feature_list, target, steps):
    train_xdata = np.array(data[feature_list])
    size = len(train_xdata)   
    x = []
    for i in range(size - steps):  # we do not have targets for the last steps of data
        sample = train_xdata[i: (i+steps)]  # a sample of steps days
        x.append(sample)
    x_train = np.array(x)
    
    y = data[target] > 0
    y = np.array(y).astype(int)
    y_train = y[steps: ]
    
    return x_train, y_train
