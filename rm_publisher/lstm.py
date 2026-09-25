import numpy as np
import torch
import yaml
import torch.nn as nn
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

def train_val_split(X, y, shuffle=True):
    if shuffle:
        random_order = torch.randperm(X.shape[0])
        X = X[random_order]
        y = y[random_order]
    split = X.shape[0] - X.shape[0] // 10
    X_train = X[:split]
    y_train = y[:split]
    X_val = X[split:]
    y_val = y[split:]
    return X_train, y_train, X_val, y_val

class DriveSequenceDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, loss_fn, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss_fn = loss_fn
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

    def train_one_epoch(self, train_dataloader, optimizer):
        self.train(True)
        total_running_loss = 0.0
        x_running_loss = 0.0
        y_running_loss = 0.0
        z_running_loss = 0.0

        for batch in tqdm(train_dataloader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            output = self(x_batch)

            x_loss = self.loss_fn(1000 * output[:, 0], 1000 * y_batch[:, 0])
            y_loss = self.loss_fn(1000 * output[:, 1], 1000 * y_batch[:, 1])
            z_loss = self.loss_fn(1000 * output[:, 2], 1000 * y_batch[:, 2])
            loss = self.loss_fn(1000*output, 1000*y_batch)

            x_loss.backward(retain_graph=True)
            y_loss.backward(retain_graph=True)
            z_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            x_running_loss += x_loss.item()
            y_running_loss += y_loss.item()
            z_running_loss += z_loss.item()

            total_running_loss += loss.item()

        x_avg_loss = x_running_loss / len(train_dataloader)
        y_avg_loss = y_running_loss / len(train_dataloader)
        z_avg_loss = z_running_loss / len(train_dataloader)
        avg_loss = total_running_loss / len(train_dataloader)

        print(f'Train Loss: {format(avg_loss, ".4f")}, X Loss: {format(x_avg_loss, ".4f")}, Y Loss: {format(y_avg_loss, ".4f")}, Z Loss: {format(z_avg_loss, ".4f")}')
        return x_avg_loss, y_avg_loss, z_avg_loss, avg_loss



    def validate_one_epoch(self, validation_dataloader):
        self.train(False)
        total_running_loss = 0.0
        x_running_loss = 0.0
        y_running_loss = 0.0
        z_running_loss = 0.0
        
        for batch in validation_dataloader:
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = self(x_batch)
                x_loss = self.loss_fn(1000*output[:, 0], 1000*y_batch[:, 0])
                y_loss = self.loss_fn(1000*output[:, 1], 1000*y_batch[:, 1])
                z_loss = self.loss_fn(1000*output[:, 2], 1000*y_batch[:, 2])
                loss = self.loss_fn(1000*output, 1000*y_batch)
                x_running_loss += x_loss.item()
                y_running_loss += y_loss.item()
                z_running_loss += z_loss.item()
                total_running_loss += loss.item()

        x_avg_loss = x_running_loss / len(validation_dataloader)
        y_avg_loss = y_running_loss / len(validation_dataloader)
        z_avg_loss = z_running_loss / len(validation_dataloader)
        avg_loss = total_running_loss / len(validation_dataloader)

        print(f'Validation Loss: {format(avg_loss, ".4f")}, X Loss: {format(x_avg_loss, ".4f")}, Y Loss: {format(y_avg_loss, ".4f")}, Z Loss: {format(z_avg_loss, ".4f")}')
        return x_avg_loss, y_avg_loss, z_avg_loss, avg_loss

def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.008
    return base_lr/(1+factor*epoch)
            
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(os.getcwd(), 'cfg/task/robomaster.yaml'), 'r') as f:
      cfg = yaml.load(f, Loader=yaml.SafeLoader)

    data_base_path = './tasks/data-8/'
    neurons = cfg['lstm']['neurons']
    layers = cfg['lstm']['layers']
    sequence = cfg['lstm']['sequence']

    look_back_array_acc = []
    y = []
    for i in range(20):
        real_data_path = f'{data_base_path}data-{i}.npz'
        base_angular_real = np.load(real_data_path)['base_angular_velocities']
        base_linear_real = np.load(real_data_path)['base_linear_velocities']
        cmd_vel = np.load(real_data_path)['cmd_vel']
    
    
        # base_vel_sim = np.stack([base_linear_sim[:, 0, 0], base_linear_sim[:, 0, 1], base_angular_sim[:, 0, 2]], axis=-1)[:50000]
        base_vel_real = np.stack([base_linear_real[:, 0], base_linear_real[:, 1], base_angular_real[:, 2] / 4.15], axis=-1)

        # X = np.stack([base_vel_sim, cmd_vel], axis=1)
        X = cmd_vel
        # X_l = np.stack([base_vel_real, cmd_vel], axis=1)
        # X = X.reshape(X.shape[0], 1, X.shape[1] * X.shape[2])
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # X_l = X_l.reshape(X_l.shape[0], 1, X_l.shape[1] * X_l.shape[2])
        if cmd_vel.shape == base_linear_real.shape:
            y.append(cmd_vel - base_vel_real)

            look_back_array = np.zeros((X.shape[0], sequence, X.shape[1] * X.shape[2]))
            for i in range(sequence):
                look_back_array[i:, i:i+1, :] = X[:X.shape[0]-i, :i+1, :]

            look_back_array_acc.append(look_back_array)

    look_back_array = np.concatenate(look_back_array_acc, axis=0)
    # look_back_array = look_back_array.reshape(look_back_array.shape[0] * look_back_array.shape[1], sequence, look_back_array.shape[-1])
    y = np.concatenate(y, axis=0)
    # y = y.reshape(y.shape[0] * y.shape[1], y.shape[-1])

    X = torch.tensor(look_back_array, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.float32)
    
    print(X.shape)
    print(y.shape)
    
    X_train, y_train, X_val, y_val = train_val_split(X, y, shuffle=True)
    train_dataset = DriveSequenceDataset(X_train, y_train)
    validation_dataset = DriveSequenceDataset(X_val, y_val)

    batch_size = 2048
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    for _, batch in enumerate(train_dataloader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    lr = 4e-2
    epochs = 250
    loss_fn = nn.MSELoss()

    model = LSTM(X.shape[-1], neurons, layers, y.shape[-1], loss_fn)
    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {trainable_params} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = {
        'total_train_loss': [],
        'x_train_loss': [],
        'y_train_loss': [],
        'z_train_loss': [],
        'total_val_loss': [],
        'x_val_loss': [],
        'y_val_loss': [],
        'z_val_loss': [],
    }
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')
        train_losses = model.train_one_epoch(train_dataloader, optimizer)
        validation_losses = model.validate_one_epoch(validation_dataloader)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Adam lr %.6f -> %.6f" % (before_lr, after_lr))
        history['total_train_loss'].append(train_losses[3])
        history['x_train_loss'].append(train_losses[0])
        history['y_train_loss'].append(train_losses[1])
        history['z_train_loss'].append(train_losses[2])
        history['total_val_loss'].append(validation_losses[3])
        history['x_val_loss'].append(validation_losses[0])
        history['y_val_loss'].append(validation_losses[1])
        history['z_val_loss'].append(validation_losses[2])
        print(f'{"*" * 100}')

    # Plot losses
    plt.figure(figsize=(12, 8))

    # Plot average loss
    plt.subplot(2, 2, 1)
    plt.plot(history['total_train_loss'], label='Total Train Loss')
    plt.plot(history['total_val_loss'], label='Total Validation Loss')
    plt.title('Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot x losses
    plt.subplot(2, 2, 2)
    plt.plot(history['x_train_loss'], label='X Train Loss')
    plt.plot(history['x_val_loss'], label='X Validation Loss')
    plt.title('X Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot y losses
    plt.subplot(2, 2, 3)
    plt.plot(history['y_train_loss'], label='Y Train Loss')
    plt.plot(history['y_val_loss'], label='Y Validation Loss')
    plt.title('Y Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot z losses
    plt.subplot(2, 2, 4)
    plt.plot(history['z_train_loss'], label='Z Train Loss')
    plt.plot(history['z_val_loss'], label='Z Validation Loss')
    plt.title('Z Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), './nn/lstm.pth')
    model2 = LSTM(X.shape[-1], neurons, layers, y.shape[-1], loss_fn, self.device)
    model2.load_state_dict(torch.load('./nn/lstm.pth'))
    print(model2)
    print(X[0].unsqueeze(0))
    res1 = model(X[0].unsqueeze(0))
    res2 = model(X[0,0].unsqueeze(0).unsqueeze(0))
    print(res1)
    print(res2)

if __name__ == '__main__':
    main()
