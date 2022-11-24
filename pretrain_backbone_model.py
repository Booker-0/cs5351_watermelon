import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

#Group1 - Watermelon Group

#LSTM Model
# LSTM PyTorch Model  URL Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#
# Citation
# LSTM (no date) LSTM - PyTorch 1.13 documentation. 
# © Copyright 2022, PyTorch Contributors. 
# Available at: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
# (Accessed: November 19, 2022).
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()


#method to load kaggle dataset
def loadKaggleDataset():

    #Dataset Citation
    #Sharanappa, S. (2021) Employee total hours timeseries prediction, 
    #Kaggle. Data files © Original Authors. 
    #Available at: https://www.kaggle.com/datasets/sunilsharanappa/employee-totalhours-timeseries-prediction?resource=download 
    #(Accessed: November 19, 2022). 
    train = pd.read_csv('Employee_Login_Logout_TimeSeries.csv')

    hours_worded = []

    for i in train["Total_Hours"]:
        hours, minutes = i.split(":")
        coverted_to_decimal_hours = int(hours) + (int(minutes)* 1.666666667 / 100)
        hours_worded.append(coverted_to_decimal_hours)

    return torch.Tensor(hours_worded)

def train():
    x = loadKaggleDataset()

    input_x = []
    input_y = []

    for day, datapoint in enumerate(x):
        input_x.append(float(datapoint))
        input_y.append(int(day))

    test_data_size = int( len(x) /2)
    train_data = input_x[:-test_data_size]
    test_data = input_x[-test_data_size:]
    train_data = torch.from_numpy(np.array(train_data))
    test_data = torch.from_numpy(np.array(test_data))

    
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    
    train_window = int(len(train_data_normalized)/3)
    print("train_window", train_window)

    #dataloader for model
    train_dataloader = []
    for i in range(len(train_data_normalized) - train_window):
        train_x = train_data_normalized[i:i+train_window]
        train_y = train_data_normalized[i+train_window:i+train_window+1]
        train_dataloader.append((train_x, train_y))
        
    #hyperparams
    epochs = 100
    learning_rate = 1e-3
    print(learning_rate)

    #training loop
    model = LSTM().cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100

    print(next(model.parameters()).device)
    loop = tqdm(range(epochs))
    for i in loop:

        total = 0
        correct = 0

        for seq, labels in train_dataloader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, model.hidden_layer_size).cuda())

            seq = torch.tensor(seq).cuda()
            labels = torch.tensor(labels).cuda()
            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            loop.set_postfix(loss=f"{single_loss.item():10.10f}")

        #per epoch accuracy validation
        total += labels.size(0)
        correct += (y_pred == labels).sum().item()

        print("Accuracy:", ((correct / total) * 100), "%")
        
    path = "checkpoints/"
    torch.save(model.state_dict(), path+"pretrain_backbone_model.pth")

if __name__ == "__main__":
    train()
    print("Trained backbone model.")