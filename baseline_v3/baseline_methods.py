import numpy as np
import copy
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from tqdm import trange


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):  # 正态
        return (data - self.mean) / self.std

    def inverse_transform(self, data):  # 就是为了传回去。。
        return (data * self.std) + self.mean


def test_error(y_predict, y_test):
    """
    Calculates MAE, RMSE, R2.
    :param y_test:
    :param y_predict.
    :return:
    """
    # print(y_predict.shape, y_test.shape)
    err = y_predict - y_test
    MAE = np.mean(np.abs(err[~np.isnan(err)]))
    s_err = err**2
    RMSE = np.sqrt(np.mean((s_err[~np.isnan(s_err)])))

    MAPE = cal_mape(y_true=y_test, y_pred=y_predict, null_val=0.0)

    return MAE, RMSE, MAPE


def cal_mape(y_true, y_pred, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def historical_average_predict(np_, period=2*18, test_ratio=0.2):
    """
    计算历史平均路线延迟。
    :param np_: numpy数组，包含每个路线的延迟数据。
    :param period: 周期，单位为样本数。默认为1天（2*18）。
    :param test_ratio: 测试数据占总数据的比例。默认为0.2。
    :return: 返回预测结果和测试数据。
    """
    # 获取数据的路线数和样本数
    n_route, n_sample = np_.shape
    # 计算测试数据和训练数据的数量
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    # 获取测试数据
    y_test = np_[:, -n_test:]
    # 深拷贝测试数据，用于存储预测结果
    y_predict = copy.deepcopy(y_test)

    # 对于每个周期内的样本，计算历史平均值并预测
    for i in range(n_train, min(n_sample, n_train + period)):
        # 获取周期内的样本下标
        inds = [j for j in range(i % period, n_train, period)]
        # 获取历史数据
        historical = np_[:, inds]
        # 对于每个路线，计算历史平均值并预测
        for k in range(n_route):
            y_predict[k, i - n_train] = historical[k,
                                                   :][~np.isnan(historical[k, :])].mean()
    # 对于每个周期之后的样本，使用上一个周期的预测结果
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict[:, start:start + size] = y_predict[:,
                                                     start - period: start + size - period]

    return y_predict, y_test


def var_predict(np_, n_forwards=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), n_lags=12, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: numpy, route x time.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_route, n_sample = np_.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = np_[:, :n_train], np_[:, n_train:]
    mean_train = np.mean(df_train[~np.isnan(df_train)])
    std_train = np.std(df_train[~np.isnan(df_train)])
    scaler = StandardScaler(mean=mean_train, std=std_train)
    data = scaler.transform(df_train)
    data[np.isnan(data)] = 0
    data = data.T
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    result = np.zeros(shape=(len(n_forwards), n_test, n_route))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        inputs = scaler.transform(np_[:, input_ind: input_ind + n_lags].T)
        inputs[np.isnan(inputs)] = 0
        prediction = var_result.forecast(inputs, max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    return scaler.inverse_transform(result), df_test


def arima_predict(np_, n_forwards=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), order=(12, 1, 0), test_ratio=0):
    """
    Multivariate time series forecasting using ARIMA model.
    :param np_: numpy, route x time.
    :param n_forwards: a tuple of horizons.
    :param order: the order of the ARIMA model (p, d, q).
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_route, n_sample = np_.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = np_[:, :n_train], np_[:, n_train:]
    mean_train = np.mean(df_train[~np.isnan(df_train)])
    std_train = np.std(df_train[~np.isnan(df_train)])
    scaler = StandardScaler(mean=mean_train, std=std_train)
    data = scaler.transform(df_train)
    data[np.isnan(data)] = 0  # Set 0
    data = data.T
    max_n_forwards = np.max(n_forwards)
    result = np.zeros(shape=(len(n_forwards), n_test, n_route))

    for r in range(n_route):
        arima_model = ARIMA(data[:, r], order=order)
        arima_result = arima_model.fit()
        for input_ind in range(n_train - max_n_forwards, n_sample - 1):
            inputs = scaler.transform(
                np_[r, input_ind: input_ind + 1].reshape(1, -1))
            inputs[np.isnan(inputs)] = 0
            prediction = arima_result.get_forecast(
                steps=n_train).predicted_mean[:max_n_forwards]
            for i, n_forward in enumerate(n_forwards):
                result_ind = input_ind - n_train + n_forward
                if 0 <= result_ind < n_test:
                    result[i, result_ind, r] = prediction[n_forward - 1]

    return scaler.inverse_transform(result), df_test


def var_predict_svr(np_, out_len=12, in_len=12, test_ratio=0.2, kernel='linear', C=1.0, epsilon=0.1):
    """
    Multivariate time series forecasting using Support Vector Regression.
    :param np_: numpy, route x time.
    :param n_forwards: output steps
    :param test_ratio: the fraction of the data used for testing.
    :param kernel: the kernel function used in SVR.
    :param C: the penalty parameter in SVR.
    :param epsilon: the epsilon parameter in SVR.
    :param n_lags: input length
    :return: [list of prediction in different horizon], dt_test
    """
    n_route, n_sample = np_.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = np_[:, :n_train], np_[:, n_train:]
    mean_train = np.mean(df_train[~np.isnan(df_train)])
    std_train = np.std(df_train[~np.isnan(df_train)])
    scaler = StandardScaler(mean=mean_train, std=std_train)
    result = []
    for route in trange(n_route):
        data = scaler.transform(df_train[route, :].T)
        data[np.isnan(data)] = 0
        X_train, Y_train = [], []
        for i in range(in_len, n_train-out_len):
            X_train.append(data[i-in_len:i])
            Y_train.append(data[i:i+out_len])
            if len(Y_train[i-in_len]) != out_len:
                print(
                    f"Anomaly at index {i}. Expected length {out_len}, but got {len(Y_train[i-in_len])}.")
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        svr_model = MultiOutputRegressor(
            SVR(kernel=kernel), n_jobs=-1)  # todo: cpu
        svr_model.fit(X_train, Y_train)
        # print(f"Fit OK in {route}!")
        X_test, Y_test = [], []
        data = scaler.transform(df_test[route, :].T)
        data[np.isnan(data)] = 0
        for i in range(in_len, n_sample - n_train - out_len):
            X_test.append(data[i-in_len:i])
            if len(X_test[i-in_len]) != in_len:
                print(
                    f"Anomaly at index {i}. Expected length {in_len}, but got {len(X_test[i-in_len])}.")
        X_test = np.array(X_test)
        Y_test = svr_model.predict(X_test)
        result.append(Y_test)
        # print(f"Predict OK in {route}!")
    result = np.array(result)
    print(f"result {result.shape}; df_test {df_test[:, in_len:].shape}")
    return scaler.inverse_transform(result), df_test[:, in_len:]
    # result: [N, T, T_out]
    # Note df_test should omit the first terms: [N, T]
