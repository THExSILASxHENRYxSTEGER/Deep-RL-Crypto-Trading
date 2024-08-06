import torch
from data_interface import Interface
from environment import ENVIRONMENT_DDPG, ENVIRONMENT_DQN
from RL_utils import DQN_AGENT, DQN_AGENT, load_q_func, load_ddpg_actor
from Data_Fetcher.global_variables import DEVICE, EPSILON, TICKERS
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plot_cum_rtrns = False

def get_rtrns_DDPG(actor_path, set_type="train"):
    intfc = Interface()
    env = ENVIRONMENT_DDPG(intfc, set_type=set_type, interval="1h", make_rtrns=False)
    actor = load_ddpg_actor(actor_path, eval=True)
    S_t = env.reset()
    D_t =  False
    actions = list()
    while not D_t:                        
        A_t = actor(torch.tensor(np.array(S_t)).float().to(DEVICE))
        A_t = A_t.detach().numpy()
        actions.append(A_t)
        S_t, _, D_t = env.step(A_t)
    actions = np.squeeze(np.array(actions)).T
    for i in range(len(actions)):
        plt.plot(range(len(actions[i])), actions[i])
        plt.show()

def get_rtrns_DQN(q_func_path, set_type="train", interval="1h"):
    intfc = Interface()
    env = ENVIRONMENT_DQN(intfc, set_type=set_type, interval=interval, make_rtrns=False)
    windows_t0 = env.windows[0]
    _, window_len = windows_t0[0].shape
    action_space = env.get_action_space() # for DDPG action space is # of currencies ie a weighting

    q_func = load_q_func(q_func_path, eval=True)

    agent = DQN_AGENT(EPSILON, action_space, q_func, DEVICE, training=False)
    S_t = env.reset()
    D_t =  False
    actions = list()
    while not D_t:                        
        A_t = agent.select_action(torch.tensor(np.array(S_t)).float().to(DEVICE), 0)
        A_t = A_t.detach().numpy()
        actions.append(A_t)
        S_t, _, D_t = env.step(A_t)
    weighting_t = np.zeros(len(TICKERS)+1)
    weighting_t[len(weighting_t)-1] = 1.0
    weights = list()
    for A_t in actions:
        weighting_t = np.zeros(len(TICKERS)+1)
        binary_A_t = "{0:b}".format(A_t)
        for i, n in enumerate(reversed(binary_A_t)):
            weighting_t[i] = int(n)
        n_invest = np.sum(weighting_t)
        if n_invest == 0:
            weighting_t[len(weighting_t)-1] = 1.0
        else:
            weighting_t /= n_invest
        weights.append(weighting_t)
    weights = np.array(weights).T

    train_data = intfc.get_set_type_dataset(set_type, interval)
    _, spcfc_train_data = intfc.get_overall_data_and_ticker_dicts(train_data)
    prcs = {key:list(spcfc_train_data[key]["close"]) for key in spcfc_train_data.keys()}
    rtrns = np.array([intfc.make_prices_to_returns(prcs[key]) for key in prcs.keys()])
    _, weights_cols = weights.shape
    _, rtrns_cols = rtrns.shape
    dffrce = rtrns_cols-weights_cols
    rtrns = rtrns[:,dffrce+1:]
    absolute_rtrns = list()
    for crncy_weighting, crncy_rtrns in zip(weights, rtrns):
        absolute_rtrns.append(crncy_weighting[1:]*crncy_rtrns)
    absolute_rtrns = np.sum(np.array(absolute_rtrns), axis=0)

    cum_rtrns = Interface.avg_weighted_cum_rtrns(weights, rtrns)
    filler = np.zeros(window_len)
    cum_rtrns = np.concatenate([filler, cum_rtrns])
    return cum_rtrns, absolute_rtrns

def sharpe_ratio_time_series(rtrns):
    sharpe_ratios = list()
    for i in range(len(rtrns)):
        sharpe_ratios.append(np.mean(rtrns[0:i+2])/np.std(rtrns[0:i+2]))
    return sharpe_ratios

def get_time_labels(set_type="train", interval="1h"):
    intrfc = Interface()
    train_data = intrfc.get_set_type_dataset(set_type, interval)
    gnrl_train_data, _ = intrfc.get_overall_data_and_ticker_dicts(train_data)
    return [datetime.fromtimestamp(ts) for ts in gnrl_train_data["open_time"]][1:] # skip last because of returns and not prices

def plot_multiple_time_series(multiple_time_series, timeline, title, series_labels, y_axis_label):
    plt.xticks(rotation=25)
    for time_series, lbl in zip(multiple_time_series, series_labels):
        plt.plot(timeline, time_series, label=lbl)
    plt.legend() 
    plt.title(title)
    plt.xlabel("Datetime")
    plt.ylabel(y_axis_label)
    plt.show()

#actor_path = "/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DDPG_CNN_90_33"
#cum_rtrns_DDPG(actor_path, set_type="test")

#q_func_path = "/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_90_66"
#cum_rtrns = cum_rtrns_DQN(q_func_path, set_type="valid")
