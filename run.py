import random
<<<<<<< HEAD
import torch
import numpy as np
import os
import torch.distributed as dist
=======
import mindspore as ms
import mindspore.numpy as np
import os
import mindspore.distributed as dist
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
from exp.exp_predict import Exp_predict

# Define global variables
global_predictions = []  # Define global variable to store predictions

class Config:
    # Parameters
    task_name = 'predict'
    is_training = 0
    model_id = 'electricity_sr_1'
    model = 'Timer'
    seed = 0
    data = 'wind'
    root_path = './dataset/wind/'
    data_path = 'wind.csv'
    features = 'M'
    freq = 't'
    checkpoints = './checkpoints/'
<<<<<<< HEAD
    ckpt_path = 'checkpoints/forecast_pth/checkpoint.pth'
=======
    ckpt_path = 'checkpoints/forecast_pth/checkpoint.ckpt'  # Changed to .ckpt for MindSpore
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
    d_model = 1024
    n_heads = 8
    e_layers = 8
    d_layers = 1
    d_ff = 2048
    factor = 3
    dropout = 0.1
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False
    num_workers = 4
    itr = 1
    train_epochs = 10
    batch_size = 2048
    patience = 3
    learning_rate = 3e-5
    des = 'Exp'
    loss = 'MSE'
    lradj = 'type1'
    use_amp = False
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0,1,2,3'
    stride = 1
    finetune_epochs = 10
    finetune_rate = 0.1
    local_rank = 0
    patch_len = 96
    subset_rand_ratio = 1
    data_type = 'custom'
    decay_fac = 0.75
    cos_warm_up_steps = 100
    cos_max_decay_steps = 60000
    cos_max_decay_epoch = 10
    cos_max = 1e-4
    cos_min = 2e-6
    use_weight_decay = 0
    weight_decay = 0.01
    use_ims = True
    output_len = 96
    train_test = 0
    is_finetuning = 0
    seq_len = 672
    label_len = 576
    pred_len = 96
    mask_rate = 0.25
    inverse = 0
<<<<<<< HEAD
=======

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
if __name__ == '__main__':
    args = Config()
    # Seed setting
    fix_seed = args.seed
    random.seed(fix_seed)
<<<<<<< HEAD
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
=======
    np.random.seed(fix_seed)
    ms.set_seed(fix_seed)  # Set MindSpore seed

    args.use_gpu = True if ms.context.get_context("device_target") == "GPU" and args.use_gpu else False
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
<<<<<<< HEAD
        gpus = torch.cuda.device_count()  # gpus per node
=======
        gpus = ms.context.get_context("device_num")  # gpus per node
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                     gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
<<<<<<< HEAD
        torch.cuda.set_device(local_rank)
=======
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

    elif args.task_name == 'predict':
        Exp = Exp_predict
    else:
        raise ValueError('task name not supported!')

    for ii in range(args.itr):
<<<<<<< HEAD
        # setting record of experiments
        setting = '{}_{}_ft{}_{}_pl{}_r{}'.format(args.task_name, args.model_id, args.finetune_rate, ii, args.patch_len, args.subset_rand_ratio)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
       # exp.train(setting)
=======
        # Setting record of experiments
        setting = '{}_{}_ft{}_{}_pl{}_r{}'.format(args.task_name, args.model_id, args.finetune_rate, ii, args.patch_len, args.subset_rand_ratio)

        exp = Exp(args)  # Set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        # exp.train(setting)
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

<<<<<<< HEAD
        torch.cuda.empty_cache()
    global_predictions2 = exp.get_predictions()
    # Print global predictions in the main function
    print("Global Predictions:", global_predictions2)
=======
        ms.context.clear_call_stack()  # Clear call stack in MindSpore

    global_predictions2 = exp.get_predictions()
    # Print global predictions in the main function
    print("Global Predictions:", global_predictions2)
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
