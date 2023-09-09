import baseline_methods as bm
import numpy as np
import pandas as pd

us_delay = np.load('./udata/udelay.npy')
print(us_delay.shape)
# Shape(N, T, F)

# us_delay_2 departure delay
us_delay_2 = us_delay[:, :, 1]

print(us_delay_2.shape)

y_predict_d, y_test_d = bm.var_predict_svr(us_delay_2, test_ratio=0.2)
np.savez("tmp_svr.npz", y_predict_d=y_predict_d, y_test_d=y_test_d)

# y_test_d = y_test_d[:-36]
time = [2, 5, 11]
out_len = 12
var_d_mae = []
var_d_rmse = []
var_d_mape = []

for i in time:
    shift = i+1
    num_sample = y_test_d.shape[-1]
    a, b, c = bm.test_error(
        y_predict_d[:, :, i], y_test_d[:, shift:num_sample-(out_len-shift)])
    # y_predict_d: [T, N]  y_test_d: [T, N], shift `i+1` slots
    var_d_mae.append(round(a, 3))
    var_d_rmse.append(round(b, 3))
    var_d_mape.append(round(c, 3))

for i, x in enumerate(zip(var_d_mae, var_d_rmse, var_d_mape)):
    print(f"Error of SVM in {(time[i]+1)}-step departure: {x}")
