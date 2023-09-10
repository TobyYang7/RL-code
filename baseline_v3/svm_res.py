import baseline_methods as bm
import numpy as np
import pandas as pd

us_delay = np.load('./udata/udelay.npy')
print(us_delay.shape)
# Shape(N, T, F)

# us_delay_1 arrival delay
us_delay_1 = us_delay[:, :, 0]
# us_delay_1 departure delay
us_delay_2 = us_delay[:, :, 1]

print(us_delay_1.shape)

# y_predict_a, y_test_a = bm.var_predict_svr(cn_delay_1, test_ratio=0.2)
# y_predict_d, y_test_d = bm.var_predict_svr(cn_delay_2, test_ratio=0.2)
y_predict_a, y_test_a = bm.var_predict_svr(us_delay_1, test_ratio=0.2)
y_predict_d, y_test_d = bm.var_predict_svr(us_delay_2, test_ratio=0.2)
np.savez("tmp_svr.npz", y_predict_a=y_predict_a, y_test_a=y_test_a,
         y_predict_d=y_predict_d, y_test_d=y_test_d)

# y_test_a = y_test_a[:-36]
time = [2, 5, 11]
out_len = 12
var_a_mae = []
var_a_rmse = []
var_a_mape = []
var_d_mae = []
var_d_rmse = []
var_d_mape = []

for i in time:
    shift = i+1
    num_sample = y_test_a.shape[-1]
    a, b, c = bm.test_error(
        y_predict_a[:, :, i], y_test_a[:, shift:num_sample-(out_len-shift)])
    # y_predict_a: [T, N]  y_test_a: [T, N], shift `i+1` slots
    var_a_mae.append(round(a, 3))
    var_a_rmse.append(round(b, 3))
    var_a_mape.append(round(c, 3))
    a, b, c = bm.test_error(
        y_predict_d[:, :, i], y_test_d[:, shift:num_sample-(out_len-shift)])
    # y_predict_a: [T, N]  y_test_a: [T, N], shift `i+1` slots
    var_d_mae.append(round(a, 3))
    var_d_rmse.append(round(b, 3))
    var_d_mape.append(round(c, 3))

for i, x in enumerate(zip(var_a_mae, var_a_rmse, var_a_mape)):
    print(f"Error of SVM in {(time[i]+1)}-step arrival: {x}")

for i, x in enumerate(zip(var_d_mae, var_d_rmse, var_d_mape)):
    print(f"Error of SVM in {(time[i]+1)}-step departure: {x}")
