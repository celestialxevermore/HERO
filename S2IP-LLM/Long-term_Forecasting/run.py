import argparse
import os
import torch
import torch.distributed as dist
# from accelerate import Accelerator, DeepSpeedPlugin
# from accelerate import DistributedDataParallelKwargs
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
torch.distributed.init_process_group(backend="nccl")


import psutil
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

p = psutil.Process()
p.cpu_affinity(range(40, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--number_variable', type=int,default=7, help='number of variable')

parser.add_argument('--root_path', type=str, default='/mnt/storage/personal/eungyeop/HERO/ETDataset/dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/mnt/storage/personal/eungyeop/ETRI_HANDOVER/checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')


# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=90, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--decay_fac', type=float, default=0.75)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--local-rank', type=int, default=-1, help='local rank for distributed training')

#parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# patching
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)
parser.add_argument('--pretrained', action='store_false',help='use finetuned GPT2',default=True)


parser.add_argument('--tokenization', type=str, default='patch', help='tokenization_method')
parser.add_argument('--training_strategy', type=str, default='none', help='training_strategy')

parser.add_argument('--add_prompt', type=int, default=0)
parser.add_argument('--add_trainable_prompt', type=int, default=0)
parser.add_argument('--prompt_length', type=int, default=1)
parser.add_argument('--sim_coef', type=float, default=0.0)
parser.add_argument('--pool_size', type=int, default=1000)
parser.add_argument('--period', type=int, default=24)
parser.add_argument('--prompt_init', type=str, default='text_prototype', help='prompt_init_type')
parser.add_argument('--trend_length', type=int, default=24, help='trend_length')
parser.add_argument('--seasonal_length', type=int, default=96, help='seasonal_length')

# custom 
parser.add_argument('--exp_info', type = str, required= True, default= "YOU SHOULD INPUT YOUR EXPERIMENTAL INFORMATION. CHECK YOUR RESULT JSON FILE CAREFULLY", help = 'Experimental detail')
parser.add_argument('--exp_protocol', type = str, required= True, default= "YOU SHOULD INPUT YOUR EXPERIMENTAL SCENARIO. CHECK YOUR RESULT JSON FILE CAREFULLY", help = 'Experimental detail')
    


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

def init_distributed():
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        

        #dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return local_rank, world_size
    else:
        print('Not using distributed mode')
        return 0, 1
    

if args.use_gpu and args.use_multi_gpu:
        args.local_rank, args.world_size = init_distributed()
else:
    args.local_rank = 0
    args.world_size = 1

# if args.is_training == 0:
#     args.use_multi_gpu = False
#     args.local_rank = 0


if args.use_gpu:
    torch.cuda.set_device(args.local_rank)

    print('Args in experiment:')
    print(args)

# args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
# if args.use_multi_gpu:
#     args.local_rank = init_distributed()
#     print(args.local_rank)
# else:
#     args.local_rank = 0


# if args.use_gpu and args.use_multi_gpu:
#     # args.dvices = args.devices.replace(' ', '')
#     # device_ids = args.devices.split(',')
#     # args.device_ids = [int(id_) for id_ in device_ids]
#     # args.gpu = args.device_ids[0]
#     torch.cuda.set_device(args.local_rank)
#     torch.distributed.init_process_group(backend='nccl')
#     args.world_size = torch.distributed.get_world_size()
# print('Args in experiment:')
# print(args)


if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast

if args.is_training:
    mses = []
    maes = []
    smapes = []
    msaes = []
    owas = []
    mapes = []

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        exp = Exp(args)  # set experiments
        # if args.use_multi_gpu:
        #     exp.model = DDP(exp.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        # best_model_path = path + '/' + 'checkpoint.pth'
        # base_path = os.path.join("/mnt/storage/personal/eungyeop/HERO/experiments", args.exp_info)
        
        # exp.model.load_state_dict(torch.load(os.path.join(base_path, f"{args.exp_protocol}", "model_checkpoint.pth")))
        # #/mnt/storage/personal/eungyeop/HERO/experiments/TEST/TEST/model_checkpoint.pth
        # if args.task_name == 'long_term_forecast':
            # mse, mae = exp.test(setting)
            # mses.append(mse)
            # maes.append(mae)
            # torch.cuda.empty_cache()  
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting)
        
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

 
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
