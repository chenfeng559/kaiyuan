import os
import time
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

class Exp_predict(Exp_Basic):
    def __init__(self, args):
        super(Exp_predict, self).__init__(args)
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.predictions = []  # Initialize an empty list to store predictions
        self.global_predictions =[] # Declare the global variable to be updated
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        self.args.device = self.device
        model = model.to(self.device)
        return model

    def load_model(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).resolve()  # Ensure compatibility across platforms
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def get_predictions(self):
        return self.global_predictions
    def test(self, setting, test=0):
        global_predictions=[]
        print('********************* begin to forecast  ***************************')
        print("Model Keys: ", self.model.state_dict().keys())
        test_data, test_loader = data_provider(self.args, flag='test')
        self.model.eval()

        folder_path = Path('./predictions') / setting
        folder_path.mkdir(parents=True, exist_ok=True)

        # Load model weights
        checkpoint_path = Path(r'C:\Users\22493\Desktop\Timer\KKD_project\forecast_wind\Large-Time-Series-Model-main\checkpoints\forecast_pth\checkpoint.pth')
        self.load_model(checkpoint_path)
        print(f'Model loaded from {checkpoint_path}')

        # Directly use the provided historical data
        csv_path = Path(r'C:\Users\22493\Desktop\Timer\KKD_project\forecast_wind\Large-Time-Series-Model-main\dataset\wind\wind.csv')
        df = pd.read_csv(csv_path)
        df = df.apply(pd.to_numeric, errors='coerce')

        df.fillna(0, inplace=True)
        df = df.iloc[:, 1:]  # Remove the first column (time column)
        historical_data = df.values  # Convert to numpy array

        # Normalization
        scaler = StandardScaler()
        historical_data = scaler.fit_transform(historical_data)

        # Convert to tensor
        historical_data = torch.tensor(historical_data).float().unsqueeze(0).to(self.device)

        # Get the shape of the data
        data_length, num_features = historical_data.shape[1], historical_data.shape[2]

        # Take the last historical data window of length seq_len
        final_input = historical_data[:, -self.args.seq_len:, :]
        final_input_mark = torch.zeros((1, self.args.seq_len, num_features)).to(self.device)

        # Construct decoder input
        dec_inp = torch.zeros((1, self.args.label_len + self.args.pred_len, final_input.shape[-1])).float().to(self.device)
        dec_inp[:, :self.args.label_len, :] = final_input[:, -self.args.label_len:, :]

        with torch.no_grad():
            # Model prediction
            if self.args.output_attention:
                outputs, _ = self.model(final_input, final_input_mark, dec_inp, final_input_mark)
            else:
                outputs = self.model(final_input, final_input_mark, dec_inp, final_input_mark)

            # Extract the last 96 time steps of the prediction
            pred = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()

        # If data is inverse normalized
        if self.args.inverse:
            pred_2d = pred.reshape(-1, num_features)
            pred_inv = scaler.inverse_transform(pred_2d)
            pred_inv = pred_inv.reshape(pred.shape)
            print('************there')
            pred = pred_inv

        print(f'Shape of preds: {pred.shape}')
        print('Prediction completed.')

        print("Predicted values for the next 96 time steps:")
        # 1. 对预测值进行扩展和逆归一化

        # Extract the last 96 time steps of the prediction
        pred = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()  # pred is now a NumPy array

        # If data is inverse normalized
        if self.args.inverse:
            pred_2d = pred.reshape(-1, num_features)
            pred_inv = scaler.inverse_transform(pred_2d)
            pred_inv = pred_inv.reshape(pred.shape)
            print('************there')
            pred = pred_inv

        print(f'Shape of preds: {pred.shape}')
        print('Prediction completed.')

        print("Predicted values for the next 96 time steps:")
        # 1. 对预测值进行扩展和逆归一化

        last_sample_prediction = pred[-1, :, -3][:96]  # No need for .detach().cpu().numpy()
        expanded_data_pred = np.zeros((165504, 15))
        expanded_data_pred[:96, -3] = last_sample_prediction  # 将 last_sample_prediction 放在最后一个维度
        inverse_transformed_pred = test_data.inverse_transform(expanded_data_pred)
        inverse_last_sample_prediction = inverse_transformed_pred[:96, -3]

        # Update the global variable
        self.global_predictions = inverse_last_sample_prediction.tolist()

        print("逆归一化后的 96 个时间步预测值: ", inverse_last_sample_prediction)
