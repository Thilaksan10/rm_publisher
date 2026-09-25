import torch
import torch.nn as nn
from pytorch_tcn import TCN
from rl_games.algos_torch.network_builder import NetworkBuilder


class RMANetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = RMANetworkBuilder.Network(name, self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):

        def __init__(self, name, params, **kwargs):
            self.load(params)
            self.action_space = kwargs['actions_num']
            self.obs_space = kwargs['input_shape']['Obs']
            self.history_size = kwargs['input_shape']['Past_obs']
            NetworkBuilder.BaseNetwork.__init__(self)

            cnn_args = params['cnn']
            cnn_args["input_shape"] = self.history_size
            net_args = {
                "units": params["mlp"]["units"],
                "activation": params["mlp"]["activation"],
                "dense_func": torch.nn.Linear
            }

            self.adaption = self._build_conv(**cnn_args)
            out_shape = self._calc_input_size(self.history_size, self.adaption)
            net_args['input_size'] = out_shape + self.obs_space[0]
            self.trunk = self._build_mlp(**net_args)
            self.value_trunk = self._build_mlp(**net_args)

            self.policy_head = torch.nn.Linear(net_args["units"][-1], self.action_space)
            self.value_head = torch.nn.Linear(net_args["units"][-1], 1)
            self.sigma = nn.Parameter(torch.zeros(self.action_space, requires_grad=True, dtype=torch.float32),
                                      requires_grad=True)

        def forward(self, obs):
            obs = obs['obs']
            current_obs = obs["Obs"]
            history = obs["Past_obs"]
            x = self.adaption(history).flatten(1)
            x = torch.cat((x, current_obs), dim=1)

            x1 = self.trunk(x)
            x2 = self.value_trunk(x)
            mus = self.policy_head(x1)
            values = self.value_head(x2)
            sigma = mus * 0 + self.sigma
            return mus, sigma, values, None

        def load(self, params):
            self.cnn = params['cnn']
            self.normalization = params.get('normalization', None)

# class MLPNetworkBuilder(NetworkBuilder):
#     def __init__(self, **kwargs):
#         NetworkBuilder.__init__(self)

#     def load(self, params):
#         self.params = params

#     def build(self, name, **kwargs):
#         net = MLPNetworkBuilder.Network(name, self.params, **kwargs)
#         return net

#     class Network(NetworkBuilder.BaseNetwork):

#         def __init__(self, name, params, **kwargs):
#             self.load(params)
#             self.action_space = kwargs['actions_num']
#             self.obs_space = kwargs['input_shape']['Obs']
#             # self.history_size = kwargs['input_shape']['Past_obs']
#             NetworkBuilder.BaseNetwork.__init__(self)

#             net_args = {
#                 "units": params["mlp"]["units"],
#                 "activation": params["mlp"]["activation"],
#                 "dense_func": torch.nn.Linear
#             }

#             net_args['input_size'] = self.obs_space[0]
#             self.trunk = self._build_mlp(**net_args)
#             self.value_trunk = self._build_mlp(**net_args)

#             self.policy_head = torch.nn.Linear(net_args["units"][-1], self.action_space)
#             self.value_head = torch.nn.Linear(net_args["units"][-1], 1)
#             self.sigma = nn.Parameter(torch.zeros(self.action_space, requires_grad=True, dtype=torch.float32),
#                                     requires_grad=True)

#         def forward(self, obs):
#             obs = obs['obs']
#             current_obs = obs["Obs"]

#             x1 = self.trunk(current_obs)
#             x2 = self.value_trunk(current_obs)
#             mus = self.policy_head(x1)
#             values = self.value_head(x2)
#             sigma = mus * 0 + self.sigma
#             return mus, sigma, values, None

#         def load(self, params):
#             self.normalization = params.get('normalization', None)

class MLPNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = MLPNetworkBuilder.Network(name, self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):

        def __init__(self, name, params, **kwargs):
            self.load(params)
            print(kwargs)
            self.action_space = kwargs['actions_num']
            self.obs_space = kwargs['input_shape']
            # self.history_size = kwargs['input_shape']['Past_obs']
            NetworkBuilder.BaseNetwork.__init__(self)

            net_args = {
                "units": params["mlp"]["units"],
                "activation": params["mlp"]["activation"],
                "dense_func": torch.nn.Linear
            }
            print(f'Obs Space: {self.obs_space}')
            net_args['input_size'] = self.obs_space[0]
            self.trunk = self._build_mlp(**net_args)
            self.value_trunk = self._build_mlp(**net_args)

            self.policy_head = torch.nn.Linear(net_args["units"][-1], self.action_space)
            self.value_head = torch.nn.Linear(net_args["units"][-1], 1)
            self.sigma = nn.Parameter(torch.zeros(self.action_space, requires_grad=True, dtype=torch.float32),
                                      requires_grad=True)

        def forward(self, obs):
            obs = obs['obs']
            current_obs = obs

            x1 = self.trunk(current_obs)
            x2 = self.value_trunk(current_obs)
            mus = self.policy_head(x1)
            values = self.value_head(x2)
            sigma = mus * 0 + self.sigma
            return mus, sigma, values, None

        def load(self, params):
            self.normalization = params.get('normalization', None)

class MLPWithRNNNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = MLPWithRNNNetworkBuilder.Network(name, self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):

        def __init__(self, name, params, **kwargs):
            self.load(params)
            self.action_space = kwargs['actions_num']
            self.obs_space = kwargs['input_shape']['Obs']
            self.history_size = kwargs['input_shape']['Past_obs']
            NetworkBuilder.BaseNetwork.__init__(self)

            rnn_args = {
                "name": params["rnn"]["name"],
                "input": self.history_size[-1], 
                "units": params["rnn"]["units"],
                "layers": params["rnn"]["layers"]
            }
            self.rnn = self._build_rnn(**rnn_args)

            net_args = {
                "units": params["mlp"]["units"],
                "activation": params["mlp"]["activation"],
                "dense_func": torch.nn.Linear
            }

            net_args['input_size'] = self.obs_space[0] + rnn_args["units"]
            self.trunk = self._build_mlp(**net_args)
            self.value_trunk = self._build_mlp(**net_args)

            self.policy_head = torch.nn.Linear(net_args["units"][-1], self.action_space)
            self.value_head = torch.nn.Linear(net_args["units"][-1], 1)
            self.sigma = nn.Parameter(torch.zeros(self.action_space, requires_grad=True, dtype=torch.float32),
                                      requires_grad=True)

        def forward(self, obs, states=None):
            obs = obs['obs']
            current_obs = obs["Obs"]
            past_obs = obs["Past_obs"]

            # Ensure the states are initialized correctly
            batch_size = past_obs.size(1)  # This should be 2048, the actual batch size
            num_layers = self.rnn.rnn.num_layers  # Number of layers in the LSTM
            hidden_size = self.rnn.rnn.hidden_size  # Hidden state size

            # print(f'[{num_layers}, {batch_size}, {hidden_size}]')

            # Initialize hidden state and cell state with correct shapes
            if states is None:
                h0 = torch.zeros(num_layers, batch_size, hidden_size).to(past_obs.device)  # hidden state
                c0 = torch.zeros(num_layers, batch_size, hidden_size).to(past_obs.device)  # cell state
                states = (h0, c0) 


            # Pass past_obs through RNN
            rnn_out, new_states = self.rnn(past_obs, states)  # rnn_out shape: (batch_size, sequence_length, hidden_size)
            rnn_out = rnn_out[:, -1, :]  # Take the last output in the sequence

            # Concatenate the RNN output with current_obs for further processing
            combined_features = torch.cat((current_obs, rnn_out), dim=-1)  # Shape: (batch_size, current_obs_size + rnn_out_size)

            # Pass combined_features through MLP
            x1 = self.trunk(combined_features)  # Use combined features (current_obs + rnn_out)
            x2 = self.value_trunk(combined_features)  # Use combined features (current_obs + rnn_out)

            mus = self.policy_head(x1)
            values = self.value_head(x2)
            sigma = mus * 0 + self.sigma
            return mus, sigma, values, new_states

        def load(self, params):
            self.normalization = params.get('normalization', None)

class RMAWithTCNNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = RMAWithTCNNetworkBuilder.Network(name, self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):

        def __init__(self, name, params, **kwargs):
            print(f'__init__')
            self.load(params)
            self.action_space = kwargs['actions_num']
            self.obs_space = kwargs['input_shape']['Obs']
            self.history_size = kwargs['input_shape']['Past_obs']
            NetworkBuilder.BaseNetwork.__init__(self)

            tcn_args = params['tcn']
            # tcn_args["input_shape"] = self.history_size
            tcn_args["num_inputs"] = self.history_size[0]
            net_args = {
                "units": params["mlp"]["units"],
                "activation": params["mlp"]["activation"],
                "dense_func": torch.nn.Linear
            }

            self.adaption = TCN(**tcn_args)
            out_shape = self._calc_input_size(self.history_size, [self.adaption])
            net_args['input_size'] = out_shape + self.obs_space[0]
            self.trunk = self._build_mlp(**net_args)
            self.value_trunk = self._build_mlp(**net_args)

            self.policy_head = torch.nn.Linear(net_args["units"][-1], self.action_space)
            self.value_head = torch.nn.Linear(net_args["units"][-1], 1)
            self.sigma = nn.Parameter(torch.zeros(self.action_space, requires_grad=True, dtype=torch.float32),
                                      requires_grad=True)

        def forward(self, obs):
            obs = obs['obs']
            current_obs = obs["Obs"]
            history = obs["Past_obs"]
            x = self.adaption(history).flatten(1)
            x = torch.cat((x, current_obs), dim=1)

            x1 = self.trunk(x)
            x2 = self.value_trunk(x)
            mus = self.policy_head(x1)
            values = self.value_head(x2)
            sigma = mus * 0 + self.sigma
            return mus, sigma, values, None

        def load(self, params):
            # This should now reference params correctly
            self.params = params
            self.tcn = params['tcn']
            self.normalization = params.get('normalization', None)