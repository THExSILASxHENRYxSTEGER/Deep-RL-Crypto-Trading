import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import os
from Data_Fetcher.global_variables import DEVICE, BATCH_SIZE, TICKERS
from random_processes import OrnsteinUhlenbeckProcess

torch.manual_seed(0)
np.random.seed(0)

############################### Replay Buffer Utilities #######################################

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device, action_space) -> None:
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = list()
        self.batch_size = batch_size
        self.action_space = action_space

    def add(self, transition):
        idx = np.random.randint(-1, len(self.buffer))
        if len(self.buffer) < self.buffer_size:
            self.buffer.insert(idx, transition)
        else:
            self.buffer[idx] = transition

    def get_batch(self, one_hot_actions=True):
        b_wndws, b_pos, b_a, b_r, b_d, b_wndws_, b_pos_ = [list() for _ in range(7)]
        for idx in np.random.randint(0, len(self.buffer), (self.batch_size,)):
            s, a, r, d, s_ = self.buffer[idx]
            wndw, pos = s
            b_wndws.append(wndw)
            b_pos.append(pos)
            b_a.append(a) # implement entire loop only with torch ie optimize
            b_r.append(r)
            b_d.append(d)
            wndw, pos = s_
            b_wndws_.append(wndw)
            b_pos_.append(pos)
        b_wndws = torch.tensor(np.array(b_wndws)).float().to(self.device)
        b_pos = torch.tensor(np.array(b_pos)).float().to(self.device)
        if one_hot_actions:
            b_a = torch.eye(self.action_space)[np.array(b_a)].float().to(self.device)
        else:
            b_a = torch.tensor(np.array(b_a)).float().to(self.device)
        b_r = torch.tensor(np.array(b_r)).float().to(self.device)
        b_d = torch.tensor(np.array(b_d)).float().to(self.device)
        b_wndws_ = torch.tensor(np.array(b_wndws_)).float().to(self.device)
        b_pos_ = torch.tensor(np.array(b_pos_)).float().to(self.device)
        return (b_wndws, b_pos), b_a, b_r, b_d, (b_wndws_, b_pos_)
    
class ReplayBuffer_DDPG(ReplayBuffer):

    def get_batch(self): 
        b_s, b_s_ = [list() for _ in range(len(TICKERS)+1)], [list() for _ in range(len(TICKERS)+1)]
        b_a, b_r, b_d = list(), list(), list()
        for idx in np.random.randint(0, len(self.buffer), (self.batch_size)):
            s, a, r, d, s_ = self.buffer[idx]
            for i, (window_s, window_s_) in enumerate(zip(s, s_)):
                b_s[i].append(window_s)
                b_s_[i].append(window_s_)
            b_a.append(a) # implement entire loop only with torch ie optimize
            b_r.append(r)
            b_d.append(d)
        b_s = [torch.tensor(np.array(crncy_windows)).float().to(self.device) for crncy_windows in b_s]
        b_a = torch.tensor(np.array(b_a)).float().to(self.device)
        b_r = torch.tensor(np.array(b_r)).float().to(self.device)
        b_d = torch.tensor(np.array(b_d)).float().to(self.device)
        b_s_ = [torch.tensor(np.array(crncy_windows_)).float().to(self.device) for crncy_windows_ in b_s_]
        return b_s, b_a, b_r, b_d, b_s_
    
class ReplayBuffer_DQN(ReplayBuffer):

    def get_batch(self): 
        b_s, b_s_ = [list() for _ in range(len(TICKERS)+1)], [list() for _ in range(len(TICKERS)+1)]
        b_a, b_r, b_d = list(), list(), list()
        for idx in np.random.randint(0, len(self.buffer), (self.batch_size)):
            s, a, r, d, s_ = self.buffer[idx]
            for i, (window_s, window_s_) in enumerate(zip(s, s_)):
                b_s[i].append(window_s)
                b_s_[i].append(window_s_)
            one_hot = np.zeros(self.action_space)
            one_hot[int(a)] = 1.0
            b_a.append(one_hot) # implement entire loop only with torch ie optimize
            b_r.append(r)
            b_d.append(d)
        b_s = [torch.tensor(np.array(crncy_windows)).float().to(self.device) for crncy_windows in b_s]
        b_a = torch.tensor(np.array(b_a)).float().to(self.device) # !!!!!!!!!!!! make one hot encoded vectors
        b_r = torch.tensor(np.array(b_r)).float().to(self.device)
        b_d = torch.tensor(np.array(b_d)).float().to(self.device)
        b_s_ = [torch.tensor(np.array(crncy_windows_)).float().to(self.device) for crncy_windows_ in b_s_]
        return b_s, b_a, b_r, b_d, b_s_

############################### Deep-Learning Utilities #######################################

class CNN(nn.Module):

    def __init__(self, cnn_layers, out_size) -> None:
        super(CNN, self).__init__()
        self.out_size = out_size
        conv_seq = list()
        for layer in cnn_layers:
            conv_seq.append(nn.Conv1d(**layer))
            conv_seq.append(nn.LeakyReLU())
        self.cnn = nn.Sequential(*conv_seq)

    def forward(self, S_t): # inputs have to be of type float
        window = S_t
        cnn_out = self.cnn(window)
        cnn_out = torch.flatten(cnn_out, start_dim=1)
        return cnn_out

    @staticmethod
    def create_conv1d_layers(in_chnls, out_chnls, out_sz, n_cnn_layers=2, kernel_size=4, kernel_div=1, cnn_intermed_chnls=1):
        cnn_layers = list()
        for i in range(n_cnn_layers):
            layer_dict = {
                "in_channels": cnn_intermed_chnls, 
                "out_channels":cnn_intermed_chnls, 
                "kernel_size": kernel_size
            }
            if i == 0:
                layer_dict["in_channels"] = in_chnls
            if i == n_cnn_layers-1:
                layer_dict["out_channels"] = out_chnls
            cnn_layers.append(layer_dict)
            out_sz = out_sz-kernel_size+1
            kernel_size = int(kernel_size/kernel_div)
            out_size = out_sz*out_chnls
        return cnn_layers, out_size
    
class LSTM(nn.Module):
    
    def __init__(self, in_sz, h_sz, n_lstm_lyrs) -> None:
        super(LSTM, self).__init__()
        self.out_size = h_sz
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(in_sz, h_sz) for _ in range(n_lstm_lyrs)])

    def forward(self, S_t, device=DEVICE):
        if len(S_t.shape) == 2:
            in_size, n_cells = S_t.shape
            S_t = S_t.reshape(1, in_size, n_cells)
        n_batches, in_size, n_cells = S_t.shape
        S_t = S_t.reshape(n_cells, n_batches, in_size)
        hx, cx = torch.zeros((n_batches, self.out_size)).to(device), torch.zeros((n_batches, self.out_size)).to(device)
        for i in range(len(self.lstm_cells)):
            hx, cx = self.lstm_cells[i](S_t[i], (hx, cx))
        return hx

############################### Deep-Q-Network Utilities #######################################

class DQN_AGENT:

    def __init__(self, eps, action_space, network, device, gamma=0.99, optimizer=Adam, loss=nn.MSELoss, training=True) -> None:
        self.eps = eps
        self.action_space = action_space
        self.device = device
        self.gamma = torch.tensor(gamma).to(self.device)
        self.training = training
        self.policy_net = network.float().to(self.device)
        self.target_net = deepcopy(network).float().to(self.device)
        self.update_target_net()
        if self.training:
            self.optimizer = optimizer(self.policy_net.parameters())
            self.loss = loss()

    def select_action(self, S_t, n_episode):
        if self.eps(n_episode) > np.random.rand() and self.training: #>
            return np.argmax(np.random.rand(self.action_space))
        else:
            S_t = self.state_to_device(S_t)
            A_t = torch.argmax(self.policy_net(S_t))
            torch.cuda.empty_cache()
            return A_t.cpu()
        
    def state_to_device(self, S_t):
        S_t = [torch.tensor(window).float().to(self.device) for window in S_t]
        return S_t

    def train(self, b_s, b_a, b_r, b_d, b_s_):
        pred = torch.sum(self.policy_net(b_s) * b_a, dim=1)
        target = b_r + (torch.ones_like(b_d)-b_d) * self.gamma * torch.max(self.target_net(b_s_), dim=1)[0]
        loss = self.loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        return loss 

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

class DQN_Q_FUNC(nn.Module):
    
    def __init__(self, crncy_encoders, action_space) -> None:
        super(DQN_Q_FUNC, self).__init__()
        self.crncy_encoders = nn.ModuleList(crncy_encoders)
        mlp_in_size = np.sum([crncy_encoder.out_size for crncy_encoder in crncy_encoders])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_space),
        )

    def forward(self, S_t, non_batch=False):
        mlp_in = list()
        for window, crncy_encoder in zip(S_t, self.crncy_encoders):
            crncy_encoding = crncy_encoder(window)
            dims = crncy_encoding.shape
            if non_batch:
                r, c = dims
                crncy_encoding = crncy_encoding.reshape(1, r*c)
            mlp_in.append(crncy_encoding)
        mlp_in = torch.hstack(mlp_in)
        mlp_out = self.mlp(mlp_in)
        return mlp_out
    
def get_base_function(q_func, q_func_components):
    if q_func == "CNN":
        cnn_layers, mlp_layers = CNN.create_conv1d_layers(*[int(param) for param in q_func_components])
        q_func = CNN(cnn_layers, mlp_layers)
    elif q_func == "LSTM":
        q_func = LSTM(*[int(param) for param in q_func_components])
    return q_func

def load_q_func(path, device=DEVICE, eval=True): # the model parameters are encoded in the name
    q_func_name = ""
    for q_func_name in os.listdir(path):
        if "DQN" in q_func_name:
            break
    assert "DQN" in q_func_name, "No model weights found"
    weights = torch.load(os.path.join(path, q_func_name), map_location=device)
    q_func_components = q_func_name.split("_")
    _ = q_func_components.pop(0)
    q_func_type = q_func_components.pop(0)
    crncy_encoders = list()
    for _ in range(len(TICKERS)+1):
        crncy_encoder = get_base_function(q_func_type, q_func_components)
        crncy_encoders.append(crncy_encoder)
    q_func = DQN_Q_FUNC(crncy_encoders, 2**(len(TICKERS)))    
    q_func.load_state_dict(weights)
    if eval:
        q_func.eval()
    return q_func

############################### DDPG Utilities #######################################

class ACTOR(nn.Module):
    
    def __init__(self, crncy_encoders, action_space) -> None:
        super(ACTOR, self).__init__()
        self.crncy_encoders = nn.ModuleList(crncy_encoders)
        mlp_in_size = np.sum([crncy_encoder.out_size for crncy_encoder in crncy_encoders])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_space)
        )
        self.final_activation = nn.Tanh() # nn.Softmax(dim=1) 

    def forward(self, S_t, non_batch=False):
        mlp_in = list()
        for window, crncy_encoder in zip(S_t, self.crncy_encoders):
            crncy_encoding = crncy_encoder(window)
            dims = crncy_encoding.shape
            if non_batch:
                r, c = dims
                crncy_encoding = crncy_encoding.reshape(1, r*c)
            mlp_in.append(crncy_encoding)
        mlp_in = torch.hstack(mlp_in)
        mlp_out = self.mlp(mlp_in)
        final = self.final_activation(mlp_out)
        return final

class CRITIC(nn.Module):
    
    def __init__(self, crncy_encoders, action_space) -> None:
        super(CRITIC, self).__init__()
        self.crncy_encoders = nn.ModuleList(crncy_encoders)
        mlp_in_size = np.sum([crncy_encoder.out_size for crncy_encoder in crncy_encoders])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_size+action_space, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_space),
        )

    def forward(self, S_t, A_t):
        mlp_in = list()
        for window, crncy_encoder in zip(S_t, self.crncy_encoders):
            crncy_encoding = crncy_encoder(window)
            mlp_in.append(crncy_encoding)
        mlp_in.append(A_t)
        mlp_in = torch.hstack(mlp_in)
        mlp_out = self.mlp(mlp_in)
        return mlp_out
    
class DDPG_AGENT:

    def __init__(self, actor, critic, eps, device, random_process, gamma=0.99, optimizer=Adam, value_loss_fn=nn.MSELoss, training=True, tau=0.001) -> None:
        #hyperparameters
        self.eps = eps
        self.device = device
        self.gamma = torch.tensor(gamma).to(self.device)
        self.training = training
        self.random_process = random_process
        self.tau = tau
        # actor model
        self.actor = actor.float().to(self.device)
        self.actor_target = deepcopy(self.actor).float().to(self.device)
        self.actor_optim  = optimizer(self.actor.parameters())
        # critic model
        self.critic = critic.float().to(self.device)
        self.critic_target = deepcopy(self.critic).float().to(self.device)
        self.critic_optim  = optimizer(self.critic.parameters())     
        self.value_loss_fn = value_loss_fn()

    def select_action(self, S_t, n_episode):
        S_t = [torch.tensor(window).float().to(self.device) for window in S_t]
        A_t = self.actor(S_t, non_batch=True).flatten().cpu().detach().numpy()
        torch.cuda.empty_cache()
        noise = self.random_process.sample() 
        noise *= int(self.training)*self.eps(n_episode)
        A_t += noise
        A_t = np.clip(A_t, -1., 1.)
        return A_t

    def train(self, b_s, b_a, b_r, b_d, b_s_):
        # critic update
        q_batch = self.critic(b_s, b_a)
        _, n_actions = b_a.shape
        b_d = torch.tile((torch.ones_like(b_d)-b_d), (n_actions, 1)).T
        target_q_batch = b_r + b_d * self.gamma * self.critic_target(b_s_, self.actor_target(b_s_))
        value_loss = self.value_loss_fn(q_batch, target_q_batch)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        # actor update
        policy_loss = -self.critic(b_s, self.actor(b_s))
        policy_loss = policy_loss.mean() 
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        torch.cuda.empty_cache()
        # target weights update
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def load_ddpg_actor(path, device=DEVICE, eval=True):
    q_func_name = ""
    for q_func_name in os.listdir(path):
        if "ACTOR" in q_func_name:
            break
    assert "ACTOR" in q_func_name, "No model weights found"
    weights = torch.load(os.path.join(path, q_func_name), map_location=device)
    folder_name = path.split("/")
    folder_name = folder_name[len(folder_name)-1].split("_")
    q_func_type = folder_name[1]
    q_func_components = q_func_name.split("_")
    _ = q_func_components.pop(0)
    crncy_encoders = list()
    for _ in range(len(TICKERS)+1):
        crncy_encoder = get_base_function(q_func_type, q_func_components)
        crncy_encoders.append(crncy_encoder)
    q_func = ACTOR(crncy_encoders, len(TICKERS))    
    q_func.load_state_dict(weights)
    if eval:
        q_func.eval()
    return q_func