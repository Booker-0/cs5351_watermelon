from flask import Flask, jsonify, request
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import torch.nn as nn
from tqdm import tqdm
from flask_cors import CORS, cross_origin

#Group1 - Watermelon Group

#DEBUG FUNCTIONS
def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True

#glob
app = Flask(__name__)
cors = CORS(app, allow_headers=["Content-Type"])

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


#method for getting predicts, based on the variables
# x             - input data
# employee name - used for inference checkpoint .pth
# project name  - used for inference checkpoint .pth
# est_hours     - used for estimate hours for a project to complete
# deadline      - how many days till the project needs to be complete
def predictData(x, employee_name, project_name, est_hours, deadline):
    #model
    checkpoint_name = str(employee_name)+"_"+str(project_name)
    checkpoint_pth = "checkpoints/"+checkpoint_name+".pth"
    model.load_state_dict(torch.load(checkpoint_pth))
    model.eval()  
    scaler = MinMaxScaler(feature_range=(-1, 1))

    #data  
    test_data = x
    test_data = torch.from_numpy(np.array(test_data))
    

    test_inputs = x.copy()
    test_data_normalized = scaler.fit_transform(test_data.reshape(-1, 1))
    test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

    test_window = int(len(test_data_normalized)/3)
    train_window = int(len(test_data_normalized)/3)

    tracked_hours = []
    tracked_hours.append(est_hours)

    #remove hours already worked
    for hours in x:
        est_hours -= hours
        tracked_hours.append(est_hours)

    #run model predictions and trackers
    days_to_complete = 0
    cumulative_hours_decay_pred_plot = []

    #infer model, using predictions to decrease the est hours to 0, 
    #with tracking to predict if task will complete in time
    while est_hours > 0:
        seq = torch.FloatTensor(test_inputs[-test_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
            seq = torch.tensor(seq)
            model_out = model(seq).item()
            
            test_inputs.append(model_out)

            #track hours
            pred = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
            pred_hours = pred[-1].item()
            est_hours -= pred_hours
            print("pred", pred[-1], " | est hour", est_hours)

            cumulative_hours_decay_pred_plot.append(est_hours)
            days_to_complete += 1

    will_complete_in_time = bool((deadline - days_to_complete) >= 0)

    return tracked_hours, cumulative_hours_decay_pred_plot, will_complete_in_time


@app.route('/train',methods=['GET', 'POST'])
@cross_origin()
def train():
    employee_name = request.args.get("employee_name")
    project_name = request.args.get("project_name")

    override_flag_for_use_pretrain = request.args.get("pretrained")


    x = request.args.getlist("data")

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

    if override_flag_for_use_pretrain == "True":
        print("Using pretrained weights from Kaggle Dataset")
        model.load_state_dict(torch.load("pretrain_backbone_model.pth"))
    else:
        print("Not using pretrained weights from Kaggle Dataset")
        
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

            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=f"{loss.item():10.10f}")

        #per epoch accuracy validation
        total += labels.size(0)
        correct += (y_pred == labels).sum().item()

        print("Accuracy:", ((correct / total) * 100), "%")
        
    path = "checkpoints/"
    torch.save(model.state_dict(), path+employee_name+"_"+project_name+".pth")
    print("Saved", employee_name+"'s "+project_name+" model.")
    json_out = '{ "Success":'+str(True)+'}'
    print("Test case 2", validateJSON(json_out))
    return json_out


@app.route('/predict',methods=['GET', 'POST'])
@cross_origin()
def predict():
    employee_name = request.args.get("employee_name")
    project_name = request.args.get("project_name")
    deadline = int(request.args.get("deadline"))
    project_est_hours = float(request.args.get("project_est_hours"))
    x = request.args.getlist("data")
    
    input_x = []
    input_y = []

    for day, datapoint in enumerate(x):
        input_x.append(float(datapoint))
        input_y.append(int(day))

    print(input_x)
    print(input_y)
    
    project_decay, project_pred_decay, will_complete_in_time = predictData(input_x, employee_name, project_name, est_hours=project_est_hours, deadline=deadline)

    if will_complete_in_time == True:
        will_complete_in_time = 1
    if will_complete_in_time == False:
        will_complete_in_time = 0

    print(input_x)
    output_preds = project_pred_decay

    final_data = len(input_x + output_preds)
    days = []
    
    for day in range(final_data):
        days.append(int(day))

    json_data = '{"input_data": '+str(project_decay)+', "predicted_data": '+str(project_pred_decay)+', "days": '+str(days)+', "deadline": '+str(deadline)+', "will_complete_by_deadline": '+str(will_complete_in_time)+'}'
    print(json_data)

    print("Test case 1", validateJSON(json_data))
    return json_data

if __name__ == '__main__':
   app.run()

