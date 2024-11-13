from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,adjust_model, format_time, save_experiment_data
from utils.metrics import metric
import torch
import torch.nn as nn
from models import  S2IPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, mase_loss, smape_loss


from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S2IPLLM': S2IPLLM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model()
    
        
        self.model = self.model.to(self.device)
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        # self.test_data, self.test_loader = self._get_data(flag='test')

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                device = torch.device(f'cuda:{self.args.local_rank}')
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        return device
      

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)

        if self.args.use_gpu and self.args.use_multi_gpu:
            model = DDP(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            
        return model

    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        
        if self.args.use_multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(data_set)
            data_loader = torch.utils.data.DataLoader(
                data_set,
                batch_size=self.args.batch_size // self.args.world_size,
                sampler=sampler,
                num_workers=self.args.num_workers,
                pin_memory = True,
                drop_last=True
            )
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='MSE':
            criterion = nn.MSELoss()
    
        elif self.args.loss=='SMAPE':
            criterion = smape_loss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(torch.bfloat16).float().to(self.device)
                # encoder - decoder
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
                        outputs,res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        
       
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):

        base_path = os.path.join("/mnt/storage/personal/eungyeop/HERO/experiments", self.args.exp_info)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_start = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            simlarity_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                           
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                        loss = self.criterion(outputs, batch_y)

    
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        
                        outputs,res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    
                    train_loss.append(loss.item())
                    simlarity_losses.append(res['simlarity_loss'].item())


                    loss += self.args.sim_coef*res['simlarity_loss']
                    
                    

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {format_time(left_time)}')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    loss.backward()
                    self.optimizer.step()
                else:
                 
                    loss.backward()
                    self.optimizer.step()

            epoch_duration = time.time() - epoch_time
            print("Epoch : {} cost time : {}".format(epoch + 1, format_time(epoch_duration)))
            train_loss = np.average(train_loss)
            sim_loss = np.average(simlarity_losses)
            vali_loss = self.vali(self.vali_data, self.vali_loader, self.criterion)

            if self.args.use_multi_gpu:
                train_loss = torch.tensor(train_loss).to(self.device)
                vali_loss = torch.tensor(vali_loss).to(self.device)
                
                dist.all_reduce(train_loss)
                dist.all_reduce(vali_loss)
                
                train_loss = train_loss.item() / self.args.world_size
                vali_loss = vali_loss.item() / self.args.world_size
                





            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Sim Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,sim_loss))
            
            
            early_stopping(vali_loss, self.model, base_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.use_multi_gpu:
                dist.barrier()

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1,self.args)
        
        if self.args.use_multi_gpu:
            dist.barrier()
        train_duration = time.time() - train_start 
        print(f"Total training time: {format_time(train_duration)}")
        model_path = os.path.join(base_path, f"{self.args.exp_protocol}")
        os.makedirs(model_path, exist_ok = True)
        torch.save(self.model.state_dict(), os.path.join(model_path,'model_checkpoint.pth'))
        return    

    def test(self, setting, test=1):

        
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            base_path = os.path.join("/mnt/storage/personal/eungyeop/HERO/experiments", self.args.exp_info)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            
            checkpoint_path = os.path.join(base_path, f"{self.args.exp_protocol}", "model_checkpoint.pth")
            state_dict = torch.load(checkpoint_path)

            if self.args.use_multi_gpu:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    self.model.module.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
        preds = []
        trues = []
        
        sim_matrix = []
        input_embedding = []
        prompted_embedding = []
        last_embedding = []


        self.model.eval()
        test_start = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs,res =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                f_dim = -1 if self.args.features == 'MS' else 0



                outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().detach().cpu().numpy()



               
                
               

                pred = outputs
                true = batch_y


                preds.append(pred)
                trues.append(true)
                if i % 100 == 0:
                    input = batch_x.float().detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    #visual_path = os.path.join(base_path, self.args.exp_info, self.args.model)
                    #visual(gt, pd, os.path.join(visual_path, str(i) + f"visual_{self.args.model}.pdf"))

        test_duration = time.time() - test_start
        print("Test duration: {}".format((format_time(test_duration))))

  
        preds = np.array(preds)
        trues = np.array(trues)
            
            
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if self.args.use_multi_gpu:
            mae = torch.tensor(mae).to(self.device)
            mse = torch.tensor(mse).to(self.device)
            rmse = torch.tensor(rmse).to(self.device)
            mape = torch.tensor(mape).to(self.device)
            mspe = torch.tensor(mspe).to(self.device)

            dist.all_reduce(mae, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse, op=dist.ReduceOp.SUM)
            dist.all_reduce(rmse, op=dist.ReduceOp.SUM)
            dist.all_reduce(mape, op=dist.ReduceOp.SUM)
            dist.all_reduce(mspe, op=dist.ReduceOp.SUM)

            mae = mae.item() / self.args.world_size
            mse = mse.item() / self.args.world_size
            rmse = rmse.item() / self.args.world_size
            mape = mape.item() / self.args.world_size
            mspe = mspe.item() / self.args.world_size
            metrics = {}
            metrics['mse'] = mse
            metrics['mae'] = mae 
            metrics['rmse'] = rmse
            metrics['mape'] = mape
            metrics['mspe'] = mspe

        if self.args.use_multi_gpu:
            dist.barrier()
        save_experiment_data(self.args, metrics, preds, trues, self.args.exp_info, self.args.exp_protocol)
        return
    
