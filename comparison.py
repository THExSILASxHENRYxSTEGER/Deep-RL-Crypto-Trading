import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from baselines import BuyAndHold
from data_interface import Interface
from macd import get_macd_cum_rtrns
from Data_Fetcher.global_variables import SET_TYPE_ENCODING, MACD_P_TIME_SCALE, MACD_Q_TIME_SCALE, HOURS_TO_LOOK_BACK
from model_metrics import get_rtrns_DQN, get_rtrns_DDPG, get_time_labels, multiple_regression, print_rtrn_mu_sigma_sharpe_ratio

# Get Buy and Hold Returns

macd_timescale = MACD_Q_TIME_SCALE+MACD_P_TIME_SCALE-4

buy_hold = BuyAndHold(interval="1h", cut_intrvl=macd_timescale)
buy_and_hold_rtrns = buy_hold.get_avg_returns()

buy_hold.print_set_avg_rtrns()
buy_hold.plot_rtrns()
#buy_hold.plot_rtrns(False)

valid_buy_hold_rtrns = buy_and_hold_rtrns[SET_TYPE_ENCODING["valid"]]
valid_time_frame = buy_hold.timeframes[SET_TYPE_ENCODING["valid"]]

valid_buy_hold_abs_rtrns = buy_hold.get_abs_rtrns()[SET_TYPE_ENCODING["valid"]]

sharpe_ratio = lambda time_series: np.mean(time_series)/np.std(time_series)

# sort all baselines of sitting with us properly with multiple grids

####################################################################### 

# search MACD over S and L

Ss = [8, 16, 32]
Ls = [24, 48, 96]

macd_cum_test_rtrns = {"S":list(), "L":list(), "final_roi":list()}
macd_test_sharpe = {"S":list(), "L":list(), "sharpe_ratio":list()}

for S in Ss:
    for L in Ls:
        macd_cum_test_rtrns["S"].append(S)
        macd_cum_test_rtrns["L"].append(L)
        macd_cum_rtrns, _ = get_macd_cum_rtrns(set_type="test", interval="1h", S=S, L=L)
        macd_cum_test_rtrns["final_roi"].append(macd_cum_rtrns[-1])
        macd_test_sharpe["sharpe_ratio"].append(sharpe_ratio(macd_cum_rtrns))

macd_test_final_roi = pd.DataFrame(macd_cum_test_rtrns).pivot('S', 'L', 'final_roi')

sns.heatmap(macd_test_final_roi, annot=True, fmt=".3f")
plt.show()

macd_test_sharpe["S"] = macd_cum_test_rtrns["S"]
macd_test_sharpe["L"] = macd_cum_test_rtrns["L"]

macd_test_sharpe = pd.DataFrame(macd_test_sharpe).pivot('S', 'L', 'sharpe_ratio')

sns.heatmap(macd_test_sharpe, annot=True, fmt=".3f")
plt.show()

max_macd_test_final_roi_indx = np.argmax(macd_cum_test_rtrns["final_roi"])

macd_valid_max_cum_rtrns, valid_macd_abs_rtrns = get_macd_cum_rtrns(set_type="valid", interval="1h", S=macd_cum_test_rtrns["S"][max_macd_test_final_roi_indx], L=macd_cum_test_rtrns["L"][max_macd_test_final_roi_indx])

##### plot the returns of both baseline methods against each other


plt.plot(valid_time_frame, valid_buy_hold_rtrns, label="Buy and Hold")  
plt.plot(valid_time_frame, macd_valid_max_cum_rtrns, label="MACD")
plt.legend()
plt.xlabel("time")
plt.ylabel("cumulative return")
plt.xticks(rotation=25)
plt.show() 

# DQN/DDPG hyperparmater visualization

models_dir = __file__.replace("/comparison.py", "/Models")

dqn_cnn_models_test_rtrns, dqn_lstm_models_test_rtrns = list(), list()
ddpg_cnn_models_test_rtrns, ddpg_lstm_models_test_rtrns = list(), list()

dqn_cnn_test_sharpe_ratio, dqn_lstm_test_sharpe_ratio = {"sharpe_ratio":list()}, {"sharpe_ratio":list()}
ddpg_cnn_test_sharpe_ratio, ddpg_lstm_test_sharpe_ratio = {"sharpe_ratio":list()}, {"sharpe_ratio":list()}

dqn_cnn_test_final_roi, dqn_lstm_test_final_roi = {"gamma":list(), "f":list(), "final_roi":list()}, {"gamma":list(), "f":list(), "final_roi":list()}
ddpg_cnn_test_final_roi, ddpg_lstm_test_final_roi = {"gamma":list(), "f":list(), "final_roi":list()}, {"gamma":list(), "f":list(), "final_roi":list()}

fs = ["30", "60", "90"]
gammas = ["99", "66", "33"]

rtrns = {"cum_rtrns" : None, "total_rtrns" : None}

def get_rtrn_dict(models_dir, rl_algortihm, q_func, f, gamma, rtrn_func, final_rois, sharpe_ratios):
    final_rois["f"].append(int(f)/100)
    final_rois["gamma"].append(int(gamma)/100)
    model_type = f"{rl_algortihm}_{q_func}_{f}_{gamma}"
    q_func_path = os.path.join(models_dir, model_type) 
    cum_rtrns, total_rtrns = rtrn_func(q_func_path, set_type="test", cut_intrvl=macd_timescale-HOURS_TO_LOOK_BACK)
    if np.isnan(cum_rtrns[-1]) or np.isnan(total_rtrns[-1]): #or len(np.where(cum_rtrns!=0)[0]) == 0:
        cum_rtrns = np.zeros_like(cum_rtrns)
        total_rtrns = np.zeros_like(total_rtrns)
        sharpe_ratio_value = 1.0
        #sharpe_ratios["sharpe_ratio"].append(0)
    else:
        sharpe_ratio_value = sharpe_ratio(cum_rtrns)
        #sharpe_ratios["sharpe_ratio"].append(sharpe_ratio_value)
    if np.isnan(sharpe_ratio_value):
        sharpe_ratio_value = 1.0
    sharpe_ratios["sharpe_ratio"].append(sharpe_ratio_value)
    rtrns = {"cum_rtrns" : cum_rtrns, "total_rtrns" : total_rtrns, "model_type":model_type}
    final_rois["final_roi"].append(cum_rtrns[-1])
    return rtrns

for f in fs:
    for gamma in gammas:
        dqn_cnn_models_test_rtrns.append(get_rtrn_dict(models_dir, "DQN", "CNN", f, gamma, get_rtrns_DQN, dqn_cnn_test_final_roi, dqn_cnn_test_sharpe_ratio))
        dqn_lstm_models_test_rtrns.append(get_rtrn_dict(models_dir, "DQN", "LSTM", f, gamma, get_rtrns_DQN, dqn_lstm_test_final_roi, dqn_lstm_test_sharpe_ratio))
        ddpg_cnn_models_test_rtrns.append(get_rtrn_dict(models_dir, "DDPG", "CNN", f, gamma, get_rtrns_DDPG, ddpg_cnn_test_final_roi, ddpg_cnn_test_sharpe_ratio))
        ddpg_lstm_models_test_rtrns.append(get_rtrn_dict(models_dir, "DDPG", "LSTM", f, gamma, get_rtrns_DDPG, ddpg_lstm_test_final_roi, ddpg_lstm_test_sharpe_ratio))

dqn_cnn_test_sharpe_ratio["f"], dqn_cnn_test_sharpe_ratio["gamma"] = dqn_cnn_test_final_roi["f"], dqn_cnn_test_final_roi["gamma"]
dqn_lstm_test_sharpe_ratio["f"], dqn_lstm_test_sharpe_ratio["gamma"] = dqn_lstm_test_final_roi["f"], dqn_lstm_test_final_roi["gamma"]
ddpg_cnn_test_sharpe_ratio["f"], ddpg_cnn_test_sharpe_ratio["gamma"] = ddpg_cnn_test_final_roi["f"], ddpg_cnn_test_final_roi["gamma"]
ddpg_lstm_test_sharpe_ratio["f"], ddpg_lstm_test_sharpe_ratio["gamma"] = ddpg_lstm_test_final_roi["f"], dqn_cnn_test_final_roi["gamma"]

# DQN test return heat diagram

dqn_cnn_best_model_idx = np.argmax(dqn_cnn_test_final_roi["final_roi"])
dqn_cnn_test_final_roi = pd.DataFrame(dqn_cnn_test_final_roi).pivot('gamma', 'f', 'final_roi')

dqn_lstm_best_model_idx = np.argmax(dqn_lstm_test_final_roi["final_roi"])
dqn_lstm_test_final_roi = pd.DataFrame(dqn_lstm_test_final_roi).pivot('gamma', 'f', 'final_roi')

vmin = min(dqn_cnn_test_final_roi.values.min(), dqn_lstm_test_final_roi.values.min())
vmax = max(dqn_cnn_test_final_roi.values.max(), dqn_lstm_test_final_roi.values.max())

fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[3,3,0.3]))

sns.heatmap(dqn_cnn_test_final_roi, annot=True, cbar=False, ax=axs[0], vmin=vmin, vmax=vmax,fmt=".3f")
sns.heatmap(dqn_lstm_test_final_roi, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmin=vmin, vmax=vmax, fmt=".3f")

axs[1].set_ylabel('')
axs[0].set_title('DQN CNN')
axs[1].set_title('DQN LSTM')

fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()

# DQN test sharpe ratio heat diagram

dqn_cnn_test_sharpe_ratio = pd.DataFrame(dqn_cnn_test_sharpe_ratio).pivot('gamma', 'f', 'sharpe_ratio')
dqn_lstm_test_sharpe_ratio = pd.DataFrame(dqn_lstm_test_sharpe_ratio).pivot('gamma', 'f', 'sharpe_ratio')

vmin = min(dqn_cnn_test_sharpe_ratio.values.min(), dqn_lstm_test_sharpe_ratio.values.min())
vmax = max(dqn_cnn_test_sharpe_ratio.values.max(), dqn_lstm_test_sharpe_ratio.values.max())

fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[3,3,0.3]))

sns.heatmap(dqn_cnn_test_sharpe_ratio, annot=True, cbar=False, ax=axs[0], vmin=vmin, vmax=vmax,fmt=".3f")
sns.heatmap(dqn_lstm_test_sharpe_ratio, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmin=vmin, vmax=vmax, fmt=".3f")

axs[1].set_ylabel('')
axs[0].set_title('DQN CNN')
axs[1].set_title('DQN LSTM')

fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()

# DDPG test return heat diagram

ddpg_cnn_best_model_idx = np.argmax(ddpg_cnn_test_final_roi["final_roi"])
ddpg_cnn_test_final_roi = pd.DataFrame(ddpg_cnn_test_final_roi).pivot('gamma', 'f', 'final_roi')

ddpg_lstm_best_model_idx = np.argmax(ddpg_lstm_test_final_roi["final_roi"])
ddpg_lstm_test_final_roi = pd.DataFrame(ddpg_lstm_test_final_roi).pivot('gamma', 'f', 'final_roi')

vmin = min(ddpg_cnn_test_final_roi.values.min(), ddpg_lstm_test_final_roi.values.min())
vmax = max(ddpg_cnn_test_final_roi.values.max(), ddpg_lstm_test_final_roi.values.max())

fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[3,3,0.3]))

sns.heatmap(ddpg_cnn_test_final_roi, annot=True, cbar=False, ax=axs[0], vmin=vmin,fmt=".3f")
sns.heatmap(ddpg_lstm_test_final_roi, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax, fmt=".3f")

axs[1].set_ylabel('')
axs[0].set_title('DDPG CNN')
axs[1].set_title('DDPG LSTM')

fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()

# DQN test sharpe ratio heat diagram

ddpg_cnn_test_sharpe_ratio = pd.DataFrame(ddpg_cnn_test_sharpe_ratio).pivot('gamma', 'f', 'sharpe_ratio')
ddpg_lstm_test_sharpe_ratio = pd.DataFrame(ddpg_lstm_test_sharpe_ratio).pivot('gamma', 'f', 'sharpe_ratio')

vmin = min(ddpg_cnn_test_sharpe_ratio.values.min(), ddpg_lstm_test_sharpe_ratio.values.min())
vmax = max(ddpg_cnn_test_sharpe_ratio.values.max(), ddpg_lstm_test_sharpe_ratio.values.max())

fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[3,3,0.3]))

sns.heatmap(ddpg_cnn_test_sharpe_ratio, annot=True, cbar=False, ax=axs[0], vmin=vmin,fmt=".3f")
sns.heatmap(ddpg_lstm_test_sharpe_ratio, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax, fmt=".3f")

axs[1].set_ylabel('')
axs[0].set_title('DDPG CNN')
axs[1].set_title('DDPG LSTM')

fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()

# cum returns performance

macd_cum_rtrns, _ = get_macd_cum_rtrns(set_type="valid", interval="1h")

dqn_cnn_valid_path = os.path.join(models_dir, dqn_cnn_models_test_rtrns[dqn_cnn_best_model_idx]["model_type"])
dqn_cnn_valid_cum_rtrns, dqn_cnn_total_cum_rtrns = get_rtrns_DQN(dqn_cnn_valid_path, set_type="valid", cut_intrvl=macd_timescale-HOURS_TO_LOOK_BACK)

dqn_lstm_valid_path = os.path.join(models_dir, dqn_lstm_models_test_rtrns[dqn_lstm_best_model_idx]["model_type"])
dqn_lstm_valid_cum_rtrns, dqn_lstm_total_cum_rtrns = get_rtrns_DQN(dqn_lstm_valid_path, set_type="valid", cut_intrvl=macd_timescale-HOURS_TO_LOOK_BACK)

ddpg_cnn_valid_path = os.path.join(models_dir, ddpg_cnn_models_test_rtrns[ddpg_cnn_best_model_idx]["model_type"])
ddpg_cnn_valid_cum_rtrns, ddpg_cnn_total_cum_rtrns = get_rtrns_DDPG(ddpg_cnn_valid_path, set_type="valid", cut_intrvl=macd_timescale-HOURS_TO_LOOK_BACK)

ddpg_lstm_valid_path = os.path.join(models_dir, ddpg_lstm_models_test_rtrns[ddpg_lstm_best_model_idx]["model_type"])
ddpg_lstm_valid_cum_rtrns, ddpg_lstm_total_cum_rtrns = get_rtrns_DDPG(ddpg_lstm_valid_path, set_type="valid", cut_intrvl=macd_timescale-HOURS_TO_LOOK_BACK)

# plot overall validation set performance

plt.plot(valid_time_frame, buy_and_hold_rtrns[SET_TYPE_ENCODING["valid"]], label="Buy and Hold")
plt.plot(valid_time_frame, macd_cum_rtrns, label="MACD")
plt.plot(valid_time_frame, dqn_cnn_valid_cum_rtrns, label="DQN CNN")
plt.plot(valid_time_frame, dqn_lstm_valid_cum_rtrns, label="DQN LSTM")
plt.plot(valid_time_frame, ddpg_cnn_valid_cum_rtrns, label="DDPG CNN")
plt.plot(valid_time_frame, ddpg_lstm_valid_cum_rtrns, label="DDPG LSTM")
plt.legend()
plt.xlabel("time")
plt.ylabel("cumulative return")
plt.xticks(rotation=25)
plt.show()

# plot regression of absolute returns 
print(valid_buy_hold_abs_rtrns.shape)
print(valid_macd_abs_rtrns.shape)
print(dqn_cnn_total_cum_rtrns.shape) 
print(dqn_lstm_total_cum_rtrns.shape) 
print(ddpg_cnn_total_cum_rtrns.shape) 
print(ddpg_lstm_total_cum_rtrns.shape)

abs_rtrns = np.array([
    valid_buy_hold_abs_rtrns,
    valid_macd_abs_rtrns,
    dqn_cnn_total_cum_rtrns, 
    dqn_lstm_total_cum_rtrns, 
    ddpg_cnn_total_cum_rtrns, 
    ddpg_lstm_total_cum_rtrns
])

labels = [
    "Buy and Hold", 
    "MACD", 
    "DQN CNN",  
    "DQN LSTM", 
    "DDPG CNN", 
    "DDPG LSTM", 
]

xlabel = "time"
ylabel = "absolute returns regression"

multiple_regression(valid_time_frame[1:], abs_rtrns, labels, xlabel, ylabel)

print_rtrn_mu_sigma_sharpe_ratio(buy_and_hold_rtrns[SET_TYPE_ENCODING["valid"]], "Buy And Hold")
print_rtrn_mu_sigma_sharpe_ratio(macd_cum_rtrns, "MACD")
print_rtrn_mu_sigma_sharpe_ratio(dqn_cnn_valid_cum_rtrns, "DQN CNN")
print_rtrn_mu_sigma_sharpe_ratio(dqn_lstm_valid_cum_rtrns, "DQN LSTM") 
print_rtrn_mu_sigma_sharpe_ratio(ddpg_cnn_valid_cum_rtrns, "DDPG CNN") 
print_rtrn_mu_sigma_sharpe_ratio(ddpg_lstm_valid_cum_rtrns, "DDPG LSTM")