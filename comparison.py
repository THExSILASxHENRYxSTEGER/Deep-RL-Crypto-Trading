import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from baselines import BuyAndHold
from data_interface import Interface
from macd import get_macd_cum_rtrns
from Data_Fetcher.global_variables import SET_TYPE_ENCODING
from model_metrics import get_rtrns_DQN, get_rtrns_DDPG, get_time_labels, plot_multiple_time_series

# Get Buy and Hold Returns

buy_hold = BuyAndHold(interval="1h")
buy_and_hold_rtrns = buy_hold.get_avg_returns()

# search MACD over S and L

Ss = [8, 16, 32]
Ls = [24, 48, 96]

macd_cum_test_rtrns = {"S":list(), "L":list(), "final_roi":list()}

for S in Ss:
    for L in Ls:
        macd_cum_test_rtrns["S"].append(S) 
        macd_cum_test_rtrns["L"].append(L)
        macd_cum_rtrns, _ = get_macd_cum_rtrns(set_type="test", interval="1h", S=S, L=L)
        macd_cum_test_rtrns["final_roi"].append(macd_cum_rtrns[-1])


macd_test_final_roi = pd.DataFrame(macd_cum_test_rtrns).pivot('S', 'L', 'final_roi')

sns.heatmap(macd_test_final_roi, annot=True, fmt=".3f")
plt.show()

max_macd_test_final_roi_indx = np.argmax(macd_cum_test_rtrns["final_roi"])

macd_valid_max_cum_rtrns, _ = get_macd_cum_rtrns(set_type="valid", interval="1h", S=macd_cum_test_rtrns["S"][max_macd_test_final_roi_indx], L=macd_cum_test_rtrns["L"][max_macd_test_final_roi_indx])



#### !!!!!!!!!!!!!!!!!!!!!!!!!! cut from all other models the q_timescale=252 hours also since then market goes down and macd evades that












# DQN/DDPG hyperparmater visualization

models_dir = __file__.replace("/comparison.py", "/Models")

dqn_cnn_models_test_rtrns, dqn_lstm_models_test_rtrns = list(), list()
ddpg_cnn_models_test_rtrns, ddpg_lstm_models_test_rtrns = list(), list()

dqn_cnn_test_final_roi, dqn_lstm_test_final_roi = {"gamma":list(), "epsilon":list(), "final_roi":list()}, {"gamma":list(), "epsilon":list(), "final_roi":list()}
ddpg_cnn_test_final_roi, ddpg_lstm_test_final_roi = {"gamma":list(), "epsilon":list(), "final_roi":list()}, {"gamma":list(), "epsilon":list(), "final_roi":list()}

epsilons = ["30", "60", "90"]
gammas = ["99", "66", "33"]

rtrns = {"cum_rtrns" : None, "total_rtrns" : None}

def get_rtrn_dict(models_dir, rl_algortihm, q_func, epsilon, gamma, rtrn_func, final_rois):
    final_rois["epsilon"].append(int(epsilon)/10)
    final_rois["gamma"].append(int(gamma)/100)
    model_type = f"{rl_algortihm}_{q_func}_{epsilon}_{gamma}"
    q_func_path = os.path.join(models_dir, model_type) 
    cum_rtrns, total_rtrns = rtrn_func(q_func_path, set_type="test")
    final_rois["final_roi"].append(cum_rtrns[-1])
    rtrns = {"cum_rtrns" : cum_rtrns, "total_rtrns" : total_rtrns, "model_type":model_type}
    return rtrns

for epsilon in epsilons:
    for gamma in gammas:
        dqn_cnn_models_test_rtrns.append(get_rtrn_dict(models_dir, "DQN", "CNN", epsilon, gamma, get_rtrns_DQN, dqn_cnn_test_final_roi))
        dqn_lstm_models_test_rtrns.append(get_rtrn_dict(models_dir, "DQN", "LSTM", epsilon, gamma, get_rtrns_DQN, dqn_lstm_test_final_roi))
        #ddpg_cnn_models_test_rtrns.append(get_rtrn_dict(models_dir, "DDPG", "CNN", epsilon, gamma, get_rtrns_DDPG, ddpg_cnn_test_final_roi))
        #ddpg_lstm_models_test_rtrns.append(get_rtrn_dict(models_dir, "DDPG", "LSTM", epsilon, gamma, get_rtrns_DDPG, ddpg_lstm_test_final_roi))

# DQN heat diagrams

dqn_cnn_best_model_idx = np.argmax(dqn_cnn_test_final_roi["final_roi"])
dqn_cnn_test_final_roi = pd.DataFrame(dqn_cnn_test_final_roi).pivot('gamma', 'epsilon', 'final_roi')

dqn_lstm_best_model_idx = np.argmax(dqn_lstm_test_final_roi["final_roi"])
dqn_lstm_test_final_roi = pd.DataFrame(dqn_lstm_test_final_roi).pivot('gamma', 'epsilon', 'final_roi')

vmin = min(dqn_cnn_test_final_roi.values.min(), dqn_lstm_test_final_roi.values.min())
vmax = max(dqn_cnn_test_final_roi.values.max(), dqn_lstm_test_final_roi.values.max())

fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[3,3,0.3]))

sns.heatmap(dqn_cnn_test_final_roi, annot=True, cbar=False, ax=axs[0], vmin=vmin,fmt=".3f")
sns.heatmap(dqn_lstm_test_final_roi, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax, fmt=".3f")

axs[1].set_ylabel('')
axs[0].set_title('DQN CNN')
axs[1].set_title('DQN LSTM')

fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()

# DQN cum returns performance


macd_cum_rtrns, _ = get_macd_cum_rtrns(set_type="valid", interval="1h")

dqn_cnn_valid_path = os.path.join(models_dir, dqn_cnn_models_test_rtrns[dqn_cnn_best_model_idx]["model_type"])
dqn_cnn_valid_cum_rtrns, dqn_cnn_total_cum_rtrns = get_rtrns_DQN(dqn_cnn_valid_path, set_type="valid")

dqn_lstm_valid_path = os.path.join(models_dir, dqn_lstm_models_test_rtrns[dqn_lstm_best_model_idx]["model_type"])
dqn_lstm_valid_cum_rtrns, dqn_lstm_total_cum_rtrns = get_rtrns_DQN(dqn_lstm_valid_path, set_type="valid")

time_labels = get_time_labels(set_type="valid")

plt.plot(time_labels, buy_and_hold_rtrns[SET_TYPE_ENCODING["valid"]], label="Buy and Hold")
plt.plot(time_labels, macd_cum_rtrns, label="MACD")
plt.plot(time_labels, dqn_cnn_valid_cum_rtrns, label="DQN CNN")
plt.plot(time_labels, dqn_lstm_valid_cum_rtrns, label="DQN LSTM")
plt.legend()
plt.xlabel("time")
plt.ylabel("cumulative return")
plt.show()

# DDPG heat diagrams


# DDPG cum returns performance

















# claculate best test models and plot cum rtrns and also pick according model and get validation cum return of said models and plot cum retruns of all models and baselines
# also get sharpe ratio and maybe R squared also t test significance like paper from sweden
# furthermore, do linear regression of modle sum rtrns


#dqn_lstm_test_final_roi = pd.DataFrame(dqn_lstm_test_final_roi).pivot('gamma', 'epsilon', 'final_roi') 
#sns.heatmap(dqn_lstm_test_final_roi, annot=True, fmt=".2f")

#ddpg_cnn_test_final_roi = pd.DataFrame(ddpg_cnn_test_final_roi).pivot('gamma', 'epsilon', 'final_roi') 
#sns.heatmap(ddpg_cnn_test_final_roi, annot=True, fmt=".2f")
#ddpg_lstm_test_final_roi = pd.DataFrame(ddpg_lstm_test_final_roi).pivot('gamma', 'epsilon', 'final_roi') 
#sns.heatmap(ddpg_lstm_test_final_roi, annot=True, fmt=".2f")













































def compare_baselines(models:dict, interval="30m", skip_set_types=["train"]): 
    buy_hold = BuyAndHold(interval=interval)
    buy_and_hold_rtrns = buy_hold.get_avg_returns()
    for i, set_type in enumerate(SET_TYPE_ENCODING.keys()):
        if set_type in skip_set_types: continue
        macd_cum_rtrns, time_steps = get_macd_cum_rtrns(set_type=set_type, interval=interval)
        model_rtrns = list()
        for key in models.keys():
            agent_type = key.split("_")[0]
            model_rtrns.append(model_cum_rtrns(models[key], agent_type, set_type, interval))
        Interface.plot_rtrns([buy_and_hold_rtrns[i], macd_cum_rtrns[1:], *model_rtrns], time_steps[1:], f"Average Cumulative Returns {set_type} set", False, ["Buy and Hold", "MACD", *list(models.keys())])

#models = {
#    #"AC_no_self_play" :      load_policy_value_func("AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1/no_pretraining"),
#    #"DQN_CNN_no_self_play":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/no_self_play"), 
#    #"DQN_CNN_self_play":     load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play"),
#    #"DQN_LSTM_no_self_play": load_q_func("DQN_LSTM_8_32_16_2_1_128_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_LSTM_8_32_16_2_1_128_1/random_actions"),
#    #"DQN_LSTM_self_play":    load_q_func("DQN_LSTM_8_7_16_2_1_128_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_LSTM_8_7_16_2_1_128_1"),
#    "DQN_CNN":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_explore_2_gamma_99"), 
#    "DQN_CNN_2":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/self_play"), 
#    }

#compare_baselines(models, "30m")