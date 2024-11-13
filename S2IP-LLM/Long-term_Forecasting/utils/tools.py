import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.fft
import json
import os 
from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from utils.metrics import metric

plt.switch_backend('agg')

def adjust_model(model, epoch, args):

    if args.training_strategy == 'progressive':
         
        if epoch >=3: 
         
         print('switch to progressive training strategy')

         for i, (name, param) in enumerate(model.named_parameters()):
            
            param.requires_grad = True  
        
        else:
            pass
    else:
        pass


def FFT_for_Period(x, k=10):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]





def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def vali(model, vali_data, vali_loader, criterion, args, device, itr):
    total_loss = []
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.eval()
    else:
        model.in_layer.eval()
        model.out_layer.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, itr)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.train()
    else:
        model.in_layer.train()
        model.out_layer.train()
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    # mases = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            
            outputs = model(batch_x[:, -args.seq_len:, :], itr)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
    elif minutes > 0:
        return f"{minutes:.0f}m {seconds:.0f}s"
    else:
        return f"{seconds:.2f}s"
    
def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj

def save_experiment_data(args, metrics, preds, trues, exp_info, exp_type):
    # exp_info 폴더 경로
    exp_info_dir = os.path.join("/mnt/storage/personal/eungyeop/HERO/experiments", exp_info)
    os.makedirs(exp_info_dir, exist_ok=True)
    
    # exp_protocol 폴더 경로
    exp_protocol_dir = os.path.join(exp_info_dir, exp_type)
    os.makedirs(exp_protocol_dir, exist_ok=True)


    common_params = {
        "batch_size": args.batch_size,
        "train_epochs": args.train_epochs,
    }
    

    model_specific_params = {
        args.model: {
            # 모델별 특정 파라미터들...
        }
    }

    # 설정 정보
    setting = {

        "data": args.data,
        #"data.preprocessing": args.preprocessing,
        "data.features": args.features,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "factor": args.factor,
        "embed": args.embed, 
        "distill": args.distil,
        "training hyper_parameters": {
            "train_epoch": args.train_epochs,
            #"distance": args.distance, 
            #"use_pos": args.use_pos,
            #"use ReverseIN": args.use_RevIN,
            "loss": args.loss,
            #"reg": args.reg,
            "moving_avg": args.moving_avg,
            #"scale_type": args.scale_type,
        },
        "Experience destination": args.des
    }

    settings = {
        "Model hyperparameter settings": setting, 
        "common_params": common_params,
        "model_specific_params": model_specific_params
    }



    model_exp_dir = os.path.join(exp_protocol_dir, args.model)
    os.makedirs(model_exp_dir, exist_ok=True)

    with open(os.path.join(model_exp_dir, "meta-setting.json"), 'w') as f:
        json.dump(settings, f, indent=4)
    
    hyper_parameters = {}
    hyper_parameters['prompt_lenth'] = args.prompt_length
    hyper_parameters['sim_coef'] = args.sim_coef
    hyper_parameters['Semantic_Space'] = args.pool_size 
    hyper_parameters['trend_length'] = args.trend_length
    hyper_parameters['seasonal_length'] = args.seasonal_length
    results = {
        "model_id": args.model_id,
        #"short_term" : args.short_term_model,
        #"long_term" : args.long_term_model,
        "metrics": metrics,
        "hyperparameters" : hyper_parameters

    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_results_{timestamp}.json"
    filepath = os.path.join(model_exp_dir, filename)

    serializable_results = convert_to_serializable(results)
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)


