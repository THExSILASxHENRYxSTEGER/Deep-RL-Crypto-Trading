import os
from baselines import BuyAndHold
from data_interface import Interface
from macd import get_macd_cum_rtrns
from Data_Fetcher.global_variables import SET_TYPE_ENCODING
from model_metrics import cum_rtrns_DQN, sharpe_ratio_time_series, plot_multiple_time_series, get_time_labels

models_dir = "/home/honta/Desktop/Thesis/prev_models_plots_and_stuff/Plots/Models" # __file__.replace("/comparison.py", "/Models")

dqn_cnn_models_test_rtrns, dqn_lstm_models_test_rtrns = list(), list()
ddpg_cnn_models_test_rtrns, ddpg_lstm_models_test_rtrns = list(), list()

explore_fracs = [0.3, 0.6, 0.9]
gammas = [0.33, 0.66, 0.99]

for explore_frac in explore_fracs:
    for gamma in gammas:
        dqn_cnn =  f"DQN_CNN_{explore_frac}_{gamma}"
        dqn_lstm = f"DQN_LSTM_{explore_frac}_{gamma}"
        ddpg_cnn = f"DDPG_CNN_{explore_frac}_{gamma}"
        ddpg_cnn = f"DDPG_LSTM_{explore_frac}_{gamma}"
        q_func_path = os.path.join(models_dir, model_dir) # get qfunpath and rtrns for all above and sove by array
        cum_rtrns, total_rtrns = cum_rtrns_DQN(q_func_path, set_type="test")
        rtrns = {
            "cum_rtrns" : cum_rtrns,
            "total_rtrns" : total_rtrns
        }
        if "DQN" and "CNN" in model_dir:
            dqn_cnn_models_test_rtrns[model_dir] = rtrns
        elif "DQN" and "LSTM" in model_dir:
            dqn_lstm_models_test_rtrns[model_dir] = rtrns
        elif "DDPG" and "CNN" in model_dir:
            ddpg_cnn_models_test_rtrns[model_dir] = rtrns
        elif "DDPG" and "LSTM" in model_dir:
            ddpg_lstm_models_test_rtrns[model_dir] = rtrns

# get best performing models and make heat diagram

explore_fracs = [0.3, 0.6, 0.9]:
gammas = [0.33, 0.66, 0.99]:

dqn_cnn_total_cum_rtrns = [for ]





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

models = {
    #"AC_no_self_play" :      load_policy_value_func("AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1/no_pretraining"),
    #"DQN_CNN_no_self_play":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/no_self_play"), 
    #"DQN_CNN_self_play":     load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play"),
    #"DQN_LSTM_no_self_play": load_q_func("DQN_LSTM_8_32_16_2_1_128_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_LSTM_8_32_16_2_1_128_1/random_actions"),
    #"DQN_LSTM_self_play":    load_q_func("DQN_LSTM_8_7_16_2_1_128_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_LSTM_8_7_16_2_1_128_1"),
    "DQN_CNN":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_explore_2_gamma_99"), 
    "DQN_CNN_2":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/self_play"), 
    }

compare_baselines(models, "30m")