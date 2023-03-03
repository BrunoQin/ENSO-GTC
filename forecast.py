import os
os.environ["LOGURU_INFO_COLOR"] = "<green>"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import datetime
import csv
import math
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from progress.spinner import MoonSpinner
from loguru import logger

from data.dataset import load_train_data, load_forecast_data
from tool.data_parallel import BalancedDataParallel
from tool.util import get_climatology, get_length_and_scaler, restore_pattern, filter_sea_and_land, get_patch, read_data, get_physics_graph, print_model_parameters
from tool.configs import args

args['model_name'] = 'Model'
save_path = './checkpoints_archive/forecast_19.pth'

def colloct_result(result):
    f = open(f'result-{datetime.datetime.today().year}-{datetime.datetime.today().month}.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(result)
    f.close()

if __name__=='__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    forecast_loader = DataLoader(dataset=load_forecast_data(), batch_size=1, shuffle=True)
    length, scaler = get_length_and_scaler(filename=args['nc_file'])
    climatology = get_climatology(filename=args['nc_file'])

    m = int(args['init_h'] / args['patch_size'])
    n = int(args['init_w'] / args['patch_size'])
    loc = filter_sea_and_land(get_patch(read_data(filename=args['raw_file'])[0], m=m, n=n), m=m, n=n, ratio=args['ratio'])
    phy_graph, _ = get_physics_graph(loc, sequence_length=args['sequence_length'])
    graph_channel = encoder_dense_config=args[args['platform']]['encoder_growth_rate'] * 2
    feature_map_size = args['patch_size']
    temporal_size = args['sequence_length'] * math.floor((math.floor((math.floor((np.sum(loc) * args['patch_size'] * args['patch_size'] - 3) / 2 + 1) - 3) / 2 + 1) - 3) / 2 + 1)
    spatial_size = args['sequence_length'] * np.sum(loc) * math.pow(math.floor((math.floor((math.floor((args['patch_size'] - 3) / 2 + 1) - 3) / 2 + 1) - 3) / 2 + 1), 2)
    embedding_size = int(temporal_size + spatial_size)
    for i in range(len(args[args['platform']]['encoder_dense_config'])):
        graph_channel += args[args['platform']]['encoder_growth_rate'] * args[args['platform']]['encoder_dense_config'][i]
        if i < 2:
            graph_channel = int(graph_channel * args[args['platform']]['encoder_theta'])
            feature_map_size = int(feature_map_size / args[args['platform']]['encoder_pool_size'][i])
    model = args['model_list'][args['model_name']](loc=loc, phy_graph=phy_graph, sequence_length=args['sequence_length'], dropout=args['dropout'],
                                                   encoder_growth_rate=args[args['platform']]['encoder_growth_rate'], encoder_bn_size=args[args['platform']]['encoder_bn_size'], encoder_theta=args[args['platform']]['encoder_theta'], encoder_dense_config=args[args['platform']]['encoder_dense_config'], encoder_pool_size=args[args['platform']]['encoder_pool_size'],
                                                   embedding_size=embedding_size, graph_channel=graph_channel * feature_map_size * feature_map_size, head=args[args['platform']]['head'], concat=args[args['platform']]['concat'], add_self_loops=args[args['platform']]['add_self_loops'], num_layers=args[args['platform']]['num_layers'], overlap=args[args['platform']]['overlap'], rank=args[args['platform']]['rank'], k=args[args['platform']]['k'], samples=args[args['platform']]['samples'],
                                                   decoder_init_feature=graph_channel, decoder_reduce_rate=args[args['platform']]['decoder_reduce_rate'], decoder_bn_size=args[args['platform']]['decoder_bn_size'], decoder_theta=args[args['platform']]['decoder_theta'], decoder_reduce_config=args[args['platform']]['decoder_reduce_config'], decoder_pool_size=args[args['platform']]['decoder_pool_size'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = BalancedDataParallel(args['batch_size'] / 2, model, dim=0).cuda()
    model.to(device)

    if args['pretrain'] and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device)['state_dict'])
        print('load model from:', save_path)

    pattern_result = []
    nino_result = []
    for step, input in enumerate(forecast_loader):
        input = input.to(device)
        model.train()
        output, _ = model(input)
        output = output.detach().cpu().numpy()
        output = restore_pattern(output.squeeze(0), loc)
        output = np.reshape(scaler.inverse_transform(np.reshape(output, (1, -1))), (args['init_h'], args['init_w']))

        month = (length + step) % 12
        pattern = output - climatology[month]
        nino34 = np.mean(output[84:96, 9:61])
        pattern_result.append(pattern)
        nino_result.append(nino34)

        template = ("-----------Forecasting {:d} now.-----------")
        logger.info(template.format(step + 1))
        del output, _
    np.savez(f"output-{args['lead_time']}.npz", data=pattern_result)
    colloct_result(np.array(nino_result).tolist())
