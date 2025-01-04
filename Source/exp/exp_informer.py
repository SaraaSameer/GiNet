from data.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from sklearn.metrics import mean_absolute_error, mean_squared_error


from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    #Exp_Basic
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    #self.model = self._build_model().to(self.device)
    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        Data = Dataset_Custom
        timeenc = 0 if args.embed != 'timeF' else 1

       
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            data_set = Data(
                root_path=args.root_path,
                data_path=args.test_data,  
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                SOC=args.SOC,
                label=args.label,
                begin=args.begin
            )
        elif flag == 'val':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            data_set = Data(
                root_path=args.root_path,
                data_path=args.train_data,  
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                SOC=args.SOC,
                label=args.label,
                begin=args.begin
            )

        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            data_set = Data(
                root_path=args.root_path,
                data_path=args.train_data, 
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                SOC=args.SOC,
                label=args.label,
                begin=args.begin
            )

        # print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss = []
        train_loss_list = []
        vali_loss_list = []
        for epoch in range(self.args.epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = torch.sqrt(criterion(pred, true))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_mean = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            train_loss_list.append(train_loss_mean)
            vali_loss_list.append(vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        loss_path = path + '/' + 'loss.csv'
        self.model.load_state_dict(torch.load(best_model_path))
        loss_data = pd.DataFrame({'train_loss': train_loss_list, 'val_loss': [None] * len(train_loss_list)})
        for i, val_loss in enumerate(vali_loss_list):
             loss_data.loc[i, 'val_loss'] = val_loss
        loss_data.to_csv(loss_path, index=False)

        return self.model


    def test(self, args, test_file):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
    
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds
        trues = trues.squeeze(-1) if trues.shape[-1] == 1 else trues
        df = pd.DataFrame({
            'Predictions': [list(pred) for pred in preds],  # Convert arrays to lists for each row
            'True_Values': [list(true) for true in trues]
        })
        df.to_csv('multistep_10In_10Out_InformerGRU_Panasonic.csv', index=False)
        flat_preds = preds.flatten()
        flat_trues = trues.flatten()
        rmse = np.sqrt(mean_squared_error(flat_trues, flat_preds))
        mae = mean_absolute_error(flat_trues, flat_preds)

        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        return       

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
     
        if self.args.SOC: 
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        else:
            dec_inp = batch_y[:, :, :-1].float().to(self.device)
            batch_x = batch_x[:, :, :-1].float().to(self.device)
      
        if self.args.use_amp:  
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:   
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
       

        return outputs, batch_y

    def Model_load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
