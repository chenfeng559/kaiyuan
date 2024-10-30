import os
import time
import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import context, Tensor
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Exp_predict(Exp_Basic):
    def __init__(self, args):
        super(Exp_predict, self).__init__(args)
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.predictions = []  # 用于存储预测结果的空列表
        self.global_predictions = []  # 声明全局变量以更新

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        self.args.device = self.device
        model.set_train(False)  # 设置模型为评估模式
        return model

    def load_model(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).resolve()  # 确保跨平台兼容性
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(self.model, param_dict)

    def get_predictions(self):
        return self.global_predictions

    def test(self, setting, test=0):
        print('********************* 开始预测 ***************************')
        print("模型键: ", self.model.parameters())

        test_data, test_loader = data_provider(self.args, flag='test')
        self.model.set_train(False)

        folder_path = Path('./predictions') / setting
        folder_path.mkdir(parents=True, exist_ok=True)

        # 加载模型权重
        checkpoint_path = Path(
            r'C:\Users\22493\Desktop\Timer\KKD_project\forecast_wind\Large-Time-Series-Model-main\checkpoints\forecast_pth\checkpoint.ckpt')
        self.load_model(checkpoint_path)
        print(f'从 {checkpoint_path} 加载模型')

        # 直接使用提供的历史数据
        csv_path = Path(
            r'C:\Users\22493\Desktop\Timer\KKD_project\forecast_wind\Large-Time-Series-Model-main\dataset\wind\wind.csv')
        df = pd.read_csv(csv_path)
        df = df.apply(pd.to_numeric, errors='coerce')

        df.fillna(0, inplace=True)
        df = df.iloc[:, 1:]  # 移除第一列（时间列）
        historical_data = df.values  # 转换为 numpy 数组

        # 归一化
        scaler = StandardScaler()
        historical_data = scaler.fit_transform(historical_data)

        # 转换为张量
        historical_data = Tensor(historical_data, ms.float32).unsqueeze(0)

        # 获取数据的形状
        data_length, num_features = historical_data.shape[1], historical_data.shape[2]

        # 取最后的历史数据窗口
        final_input = historical_data[:, -self.args.seq_len:, :]
        final_input_mark = Tensor(np.zeros((1, self.args.seq_len, num_features)), ms.float32)

        # 构造解码器输入
        dec_inp = Tensor(np.zeros((1, self.args.label_len + self.args.pred_len, final_input.shape[-1])), ms.float32)
        dec_inp[:, :self.args.label_len, :] = final_input[:, -self.args.label_len:, :]

        with ms.no_grad():
            # 模型预测
            if self.args.output_attention:
                outputs, _ = self.model(final_input, final_input_mark, dec_inp, final_input_mark)
            else:
                outputs = self.model(final_input, final_input_mark, dec_inp, final_input_mark)

            # 提取最后的预测时间步
            pred = outputs[:, -self.args.pred_len:, :].asnumpy()

        # 如果需要逆归一化
        if self.args.inverse:
            pred_2d = pred.reshape(-1, num_features)
            pred_inv = scaler.inverse_transform(pred_2d)
            pred_inv = pred_inv.reshape(pred.shape)
            print('应用了逆归一化。')
            pred = pred_inv

        print(f'预测值的形状: {pred.shape}')
        print('预测完成。')

        print("未来时间步的预测值:")

        # 提取最后的预测样本
        last_sample_prediction = pred[-1, :, -3][:96]
        expanded_data_pred = np.zeros((165504, 15))
        expanded_data_pred[:96, -3] = last_sample_prediction
        inverse_transformed_pred = test_data.inverse_transform(expanded_data_pred)
        inverse_last_sample_prediction = inverse_transformed_pred[:96, -3]

        # 更新全局变量
        self.global_predictions = inverse_last_sample_prediction.tolist()

        print("逆归一化后的未来时间步预测值: ", inverse_last_sample_prediction)
