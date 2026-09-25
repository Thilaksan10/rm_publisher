import torch

class FrameStack:

    def __init__(self, num_envs, frame_stack_dim, obs_dim, device):

        self.num_envs = num_envs
        self.frame_stack_dim = frame_stack_dim
        self.obs_dim = obs_dim

        self.device = device

        self.obs_history = torch.zeros((self.num_envs, self.frame_stack_dim, self.obs_dim), device=self.device, dtype=torch.float)
        self.history_pointer = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
        self.envs_aranged = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    def add_obs_history(self, obs):
        overflow = self.history_pointer >= self.frame_stack_dim
        if overflow.any():
            self.obs_history[overflow, :-1] = self.obs_history[overflow, 1:]
        self.history_pointer = self.history_pointer.clamp(0, self.frame_stack_dim - 1)
        self.obs_history[self.envs_aranged, self.history_pointer] = obs
        self.history_pointer += 1

    def return_obs(self, flatten=False):
        if flatten:
            return self.obs_history.flatten(1)
        else:
            return self.obs_history

    def reset_buffer(self, env_ids):
        self.obs_history[env_ids] = 0
        self.history_pointer[env_ids] = 0