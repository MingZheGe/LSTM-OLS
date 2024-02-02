import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense ,Dropout
import re
from datetime import timedelta
MarketIndex = "上证能源"
import matplotlib.pyplot as plt


df_data = pd.read_csv("成品油调价记录/上调记录.csv")
df_data['时间'] = pd.to_datetime(df_data['时间'], format='%Y年%m月%d日').dt.date

print(df_data['时间'][0])
event_datas = []

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

for te in df_data['时间']:
    event_datas.append(te)
event_datas = [str(date) for date in df_data['时间']]
print(event_datas)

event_Window = 20

for event_data in event_datas:
    df = pd.read_csv(MarketIndex)
    df = df.set_index('Date')
    if event_data not in df.index:
        # If not, find the next available date
        event_data = df.index[df.index > event_data].min()
    selected_row = df.loc[event_data]
    result33 = df.loc[:selected_row.name].tail(event_Window + 1)
    first_row2 = result33.head(1)
    selected_index = df.index.get_loc(event_data)
    next_row = df.iloc[selected_index + event_Window]
    end_date_window = next_row.name
    start_date_window = first_row2.index[0]
    print(end_date_window)
    # 获取该行的前150行数据
    result = df.loc[:selected_row.name].tail(151)
    result2 = df.loc[:selected_row.name].tail(21)
    first_row = result.head(1)
    first_row2 = result2.head(1)

    # 设置日期范围
    start_date = first_row.index[0]  # 计算日均收益
    end_date = first_row2.index[0]

    import os

    # 设置文件夹路径
    folder_path = "成品油2019-2023"

    # 获取文件夹中所有 CSV 文件的文件名
    file_names = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # 创建一个空的 DataFrame 用于存储结果
    result_df = pd.DataFrame(columns=['Date'])

    # 遍历每个 CSV 文件
    for file_name in file_names:
        '''
        if file_name == "鲁润股份600157.csv":
            continue
        '''
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        stock_code = re.search(r'\d+', file_name).group()
        print(stock_code)

        df = pd.read_csv(file_path)  # 个股
        df['Date'] = pd.to_datetime(df['Date'])
        # 计算涨跌幅
        df['Daily_Return'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)

        df2 = pd.read_csv(MarketIndex)  # 市场指标
        df2['Date'] = pd.to_datetime(df2['Date'])

        # 使用日期范围进行筛选
        df2 = df2[(df2['Date'] >= start_date) & (df2['Date'] <= end_date)]

        # 计算涨跌幅
        df2['Daily_Return'] = (df2['Close'] - df2['Close'].shift(1)) / df2['Close'].shift(1)
        # 打印结果
        df = df[(df['Date'] > start_date) & (df['Date'] <= end_date)]
        df2 = df2[(df2['Date'] > start_date) & (df2['Date'] <= end_date)]
        # print(df[['Date', 'Close', 'Daily_Return']])
        # print(df2[['Date', 'Close', 'Daily_Return']])

        merged_data = pd.merge(df2[['Date', 'Daily_Return']], df[['Date', 'Daily_Return']], on='Date')
        print(merged_data)
        # print(merged_data)
        y = merged_data['Daily_Return_y']
        X = sm.add_constant(merged_data['Daily_Return_x'])
        OLSmodel = sm.OLS(y, X).fit()

        # 打印回归结果
        print(OLSmodel.summary())
        print(OLSmodel.params)
        print(OLSmodel.params.Daily_Return_x)  # k
        print(OLSmodel.params.const)  # b

        X_train = merged_data[['Daily_Return_x']]
        y_train = merged_data[['Daily_Return_y']]

        #步长为2
        print(X_train)
        X_train = np.hstack((X_train[:-1], X_train[1:]))
        print(X_train)
        X_train = np.reshape(X_train, (X_train.shape[0], 2, 1))
        y_train = y_train.to_numpy().reshape(-1, 1)
        y_train = y_train[:-1]


        df3 = pd.read_csv(file_path)  # 个股
        df3['Date'] = pd.to_datetime(df3['Date'])
        df3['Daily_Return'] = (df3['Close'] - df3['Close'].shift(1)) / df3['Close'].shift(1)
        start_date_window_datetime = pd.to_datetime(start_date_window)
        df3 = df3[(df3['Date'] >= start_date_window_datetime - timedelta(days=1)) & (df3['Date'] <= end_date_window)]

        df4 = pd.read_csv(MarketIndex)  # 市场指标
        df4['Date'] = pd.to_datetime(df4['Date'])
        df4['Daily_Return'] = (df4['Close'] - df4['Close'].shift(1)) / df4['Close'].shift(1)
        df4 = df4[(df4['Date'] >= start_date_window) & (df4['Date'] <= end_date_window)]
        # print(df3)
        print("这是df4")
        print(df4)

        merged_data2 = pd.merge(df3[['Date', 'Daily_Return']], df4[['Date', 'Daily_Return']], on='Date', how="right")
        print("***")
        print(merged_data2)
        # 使用 rename 方法为 'Daily_Return_x' 列重命名，并将结果重新赋值给 'Daily_Return_x'
        merged_data2['实际收益率'] = merged_data2['Daily_Return_x'].rename('实际收益率')
        print(merged_data2['Daily_Return_y'])

        merged_data2['线性回归预期收益率'] = merged_data2['Daily_Return_y'] * OLSmodel.params.Daily_Return_x + OLSmodel.params.const
        print(merged_data2['Daily_Return_y'])

        X_test = merged_data2[['Daily_Return_x']]
        y_test = merged_data2[['Daily_Return_y']]

        X_test = np.hstack((X_test[:-1], X_test[1:]))
        X_test = np.reshape(X_test, (X_test.shape[0], 2, 1))
        '''
        y_test = y_test.to_numpy().reshape(-1, 1)
        y_test = y_test[:-1]
        '''
        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(
            140,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(140, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, epochs=800, batch_size=64, verbose=1)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)
        print(y_pred)
        print(merged_data2['实际收益率'])
        merged_data2 = merged_data2.drop(merged_data2.index[-1])
        merged_data2['lstm预期收益率'] = y_pred
        print(merged_data2['lstm预期收益率'])
        print(merged_data2['实际收益率'])
        print(merged_data2['线性回归预期收益率'])
        print(merged_data2['lstm预期收益率'])

        merged_data2['异常收益率'] = merged_data2['实际收益率'] - merged_data2[
            'lstm预期收益率']  # 这里是日常－预期，merged_data2['实际收益率']是预期收益率，
        # result_df = pd.concat([result_df, merged_data2[['Date', '实际收益率', 'Daily_Return_x', '异常收益率']]],ignore_index=True)
        result_df["Date"] = merged_data2['Date']
        file_name = os.path.splitext(file_name)[0]
        result_df[file_name] = merged_data2['异常收益率']
        # 绘制折线图
        plt.figure(figsize=(10, 6))

        # 绘制 Daily_Return_x
        plt.plot(merged_data2.index, merged_data2['实际收益率'], label='actual', color='red')

        # 绘制 实际收益率
        plt.plot(merged_data2.index, merged_data2['线性回归预期收益率'], label='olsPredict', color='blue')

        plt.plot(merged_data2.index, merged_data2['lstm预期收益率'], label='lstmPredict', color='green')

        # 添加标题和标签
        plt.title('predict vs actual')
        plt.xlabel('日期')
        plt.ylabel('收益率')
        plt.legend()

        # 显示图形
        plt.show()
        break;
    break;



