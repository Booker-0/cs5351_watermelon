# cs5351_watermelon
group project code

Machine Learning Branch

- index.html is a front end resource for inferncing the ML API route /predict

- machine_learning_api.py contains:
    - the Flask API with routes /predict & /train
    - the LSTM model
    - training and prediction methods for the model
    
- pretrain_backbone_model.py is used to pretrain the LSTM in the Employee_Login_Logout_TimeSeries 
  dataset from Kaggle. This .py file trains the pretrain_backbone_model.pth checkpoint file.
  The backbone is optional in training user based checkpoints. By default, the backbone is not pretrained
  on this dataset.


Running:
To launch the API, run the machine_learning_api.py. 
This will launch the API at the url http://localhost:5000, with the routes /train and /predict

