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

class FCNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 10)
        self.layer3 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
if __name__ == '__main__':

    data = np.load('./CADETProcess/nn/data/data_100.npy')
    #data[:,2:] = normalize(data[:,2:], axis=1)

    dataset = CADETDataset(data, 2)

    trainset, testset = torch.utils.data.random_split(dataset, [80, 20])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, drop_last=True)
        
    model = FCNN(901, 2)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), 0.01)#, weight_decay=0.01)


    # hyperparameters
    nb_epochs = 30  

    # Training Loop
    for epoch in range(nb_epochs):
        for x, target in trainloader:
            out = model(x)
            error = loss(out, target)
            optim.zero_grad()
            error.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            optim.step()

        # validation every epoch for cool graphics
        with torch.no_grad():

            total_error = 0
            num_examples = 0
            for x, target in testloader:
                out = model(x)
                error = loss(out, target)
                total_error += error
                num_examples += 1
            av_error = total_error / num_examples
            print(av_error.item())

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

    torch.save(model.state_dict(), './CADETProcess/nn/model_weights/model100.sdict')