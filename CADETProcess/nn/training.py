import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sklearn.preprocessing import normalize

class CADETDataset(torch.utils.data.Dataset):

    def __init__(self, data, nb_vars):
        super().__init__()
        self.data = torch.from_numpy(data[:,nb_vars:])
        self.data = self.data.float()

        self.target = torch.from_numpy(data[:, :nb_vars])
        self.target = self.target.float()

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.data.size(0)
    
class CADETSelfSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, data, nb_vars, mean, std_dev):
        super().__init__()
        self.data = torch.from_numpy(data[:,nb_vars:])
        std_dev[std_dev == 0] = 1.0
        self.data = (self.data - mean) / std_dev
        self.data = self.data.float()

    def __getitem__(self, item):
        return self.data[item], self.data[item]

    def __len__(self):
        return self.data.size(0)

class FCNN(nn.Module):

    def __init__(self, input_dim, output_dim, mean, std_dev):
        super().__init__()
        self.mean = mean
        self.std_dev = std_dev
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 10)
        self.layer4 = nn.Linear(10, output_dim)

    def forward(self, x):
        self.std_dev[self.std_dev == 0] = 1.0
        x = (x - self.mean) / self.std_dev
        x = F.relu(self.layer1(x.float()))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path + '.sdict')
        with open(path + '.args', 'wb') as f:
            pickle.dump((self.mean, self.std_dev), f)

    @staticmethod
    def load(input_dim, output_dim, path):
        with open(path + '.args', 'rb') as f:
            mean, std_dev = pickle.load(f)

        model = FCNN(input_dim, output_dim, mean, std_dev)

        state_dict = torch.load(path + '.sdict')
        model.load_state_dict(state_dict)
        return model
    
class AutoEncoder(nn.Module):

    def __init__(self, input_dim, param_dim):
        super().__init__()
        self.enc0 = nn.Linear(input_dim, 100)
        self.enc1 = nn.Linear(100, 100)
        self.enc2 = nn.Linear(100, param_dim)

    def forward(self, x):
        x = F.relu(self.enc0(x))
        x = F.relu(self.enc1(x))
        return self.enc2(x)

class AutoDecoder(nn.Module):

    def __init__(self, input_dim, param_dim):
        super().__init__()
        self.dec0 = nn.Linear(param_dim, 100)
        self.dec1 = nn.Linear(100, 100)
        self.dec2 = nn.Linear(100, input_dim)

    def forward(self, x):
        x = F.relu(self.dec0(x))
        x = F.relu(self.dec1(x))
        return self.dec2(x)
    
class AutoEncoderDecoder(nn.Module):

    def __init__(self, input_dim, param_dim):
        super().__init__()
        self.enc = AutoEncoder(input_dim, param_dim)
        self.dec = AutoDecoder(input_dim, param_dim)

    def forward(self, x):
        x = self.enc(x)
        return self.dec(x)
    
    def encode(self, x):
        return self.enc(x)
    

def train_ae(input_dim, output_dim, model_path, data_path):

    data = np.load(data_path)
    mean = np.mean(data[:,output_dim:], axis=0)
    std_dev = np.std(data[:, output_dim:], axis=0)

    ae_dataset = CADETSelfSupervisedDataset(data, 2, mean, std_dev)

    ae_trainset, ae_testset = torch.utils.data.random_split(ae_dataset, [800, 200])
    ae_trainloader = torch.utils.data.DataLoader(ae_trainset, batch_size=10, shuffle=True, drop_last=True)
    ae_testloader = torch.utils.data.DataLoader(ae_testset, batch_size=10, shuffle=False, drop_last=True)
        
    ae = AutoEncoderDecoder(901, 2)
    ae_loss = nn.MSELoss()
    ae_optim = torch.optim.Adam(ae.parameters(), 0.001, weight_decay=0.1)

    # hyperparameters
    nb_epochs = 100

    # Training Loop
    for epoch in range(nb_epochs):
        train_loss = 0
        for x, target in ae_trainloader:
            out = ae(x)
            error = ae_loss(out, target)
            train_loss += error
            ae_optim.zero_grad()
            error.backward()
            ae_optim.step()

        # validation every epoch for cool graphics
        with torch.no_grad():

            total_error = 0
            num_examples = 0
            for x, target in ae_testloader:
                out = ae(x)
                error = ae_loss(out, target)
                total_error += error
                num_examples += 1
            av_error = total_error / num_examples
            print(f'{train_loss} | {av_error.item()}')

    with torch.no_grad():

        total_error = 0
        num_examples = 0
        for x, target in ae_testloader:
            out = ae(x)
            error = ae_loss(out, target)
            total_error += error
            num_examples += 1
        av_error = total_error / num_examples
        print(av_error)
    

def train_fcnn(input_dim, output_dim, model_path, data_path):

    # hyperparameters
    nb_epochs = 300
    train_test_split = 0.8
    batch_size = 10
    lr = 0.0001

    data = np.load(data_path)
    mean = np.mean(data[:,output_dim:], axis=0)
    std_dev = np.std(data[:, output_dim:], axis=0)

    dataset = CADETDataset(data, output_dim)

    trainset, testset = torch.utils.data.random_split(dataset, [int(train_test_split*len(dataset)), int(len(dataset) - train_test_split*len(dataset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
        
    model = FCNN(input_dim, output_dim, mean, std_dev)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr)#, weight_decay=0.1)

    # Training Loop
    for epoch in range(nb_epochs):
        train_loss = 0
        for x, target in trainloader:
            out = model(x)
            error = loss(out, target)
            train_loss += error
            optim.zero_grad()
            error.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            optim.step()

        # validation every epoch for cool graphics
        # with torch.no_grad():

        #     total_error = 0
        #     num_examples = 0
        #     for x, target in testloader:
        #         out = model(x)
        #         error = loss(out, target)
        #         total_error += error
        #         num_examples += 1
        #     av_error = total_error / num_examples
        #     print(f'{train_loss} | {av_error.item()}')
        if epoch % 10 == 9:
            print(train_loss)

    with torch.no_grad():

        total_error = 0
        num_examples = 0
        for x, target in testloader:
            out = model(x)
            error = loss(out, target)
            total_error += error
            num_examples += 1
        av_error = total_error / num_examples
        print(av_error)
    
    model.save(model_path)

    
if __name__ == '__main__':

    train_fcnn(901, 2, './CADETProcess/nn/model_weights/model1000', './CADETProcess/nn/data/data_1000.npy')