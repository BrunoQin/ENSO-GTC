import os
os.environ["LOGURU_INFO_COLOR"] = "<green>"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
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

from data.dataset import load_train_data
from tool.data_parallel import BalancedDataParallel
from tool.util import filter_sea_and_land, get_patch, read_data, get_physics_graph, print_model_parameters
from tool.configs import args

args['model_name'] = 'Model'
save_path = os.path.join(r'./file', args['model_name'] + '.pth')

if __name__=='__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(dataset=load_train_data(), batch_size=args['batch_size'], shuffle=True)

    m = int(args['init_h'] / args['patch_size'])
    n = int(args['init_w'] / args['patch_size'])
    sequence_length = args['sequence_length']
    loc = filter_sea_and_land(get_patch(read_data(filename=args['raw_file'])[0], m=m, n=n), m=m, n=n, ratio=args['ratio'])
    phy_graph, _ = get_physics_graph(loc, sequence_length=sequence_length)
    graph_channel = encoder_dense_config=args[args['platform']]['encoder_growth_rate'] * 2
    feature_map_size = args['patch_size']
    temporal_size = sequence_length * math.floor((math.floor((math.floor((np.sum(loc) * args['patch_size'] * args['patch_size'] - 3) / 2 + 1) - 3) / 2 + 1) - 3) / 2 + 1)
    spatial_size = sequence_length * np.sum(loc) * math.pow(math.floor((math.floor((math.floor((args['patch_size'] - 3) / 2 + 1) - 3) / 2 + 1) - 3) / 2 + 1), 2)
    embedding_size = int(temporal_size + spatial_size)
    for i in range(len(args[args['platform']]['encoder_dense_config'])):
        graph_channel += args[args['platform']]['encoder_growth_rate'] * args[args['platform']]['encoder_dense_config'][i]
        if i < 2:
            graph_channel = int(graph_channel * args[args['platform']]['encoder_theta'])
            feature_map_size = int(feature_map_size / args[args['platform']]['encoder_pool_size'][i])
    model = args['model_list'][args['model_name']](loc=loc, phy_graph=phy_graph, sequence_length=sequence_length, dropout=args['dropout'],
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
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    criterion_1 = nn.MSELoss(reduction='mean').to(device)
    criterion_2 = nn.L1Loss(reduction='mean').to(device)

    logger.add(f"./train.log", enqueue=True)

    for epoch in range(args['n_epochs']):
        loss_epoch = 0
        sample = 0
        acc_time = 0
        loss_mse = 0
        loss_mae = 0
        loss_elbo = 0
        for step, (input, output) in enumerate(train_loader):
            start = time.process_time()

            optimizer.zero_grad()

            input = input.to(device)
            output = output.to(device)
            model.train()

            pred, elbo = model(input)

            mse = criterion_1(pred, output)
            mae = criterion_2(pred, output)
            loss_mse += mse
            loss_mae += mae

            loss_elbo += elbo
            loss = (mse + mae + elbo / 1e4).sum()

            loss_epoch += loss
            sample += 1

            elapsed = (time.process_time() - start)
            acc_time += elapsed
            if step % args['record'] == 0:
                template = ("epoch {} - step {}: loss is {:1.5f}, inclding mse {:1.5f}, mae {:1.5f} and elbo {:1.5f}, ({:1.2f}s/record)")
                logger.info(template.format(epoch, step, np.sum(loss_epoch.cpu().detach().numpy())/(step+1), np.sum(loss_mse.cpu().detach().numpy())/(step+1), np.sum(loss_mae.cpu().detach().numpy())/(step+1), np.sum(loss_elbo.cpu().detach().numpy())/((step+1)*1e4), acc_time/args['record']))
                acc_time = 0

            loss.backward()
            optimizer.step()

        template = ("-----------epoch {} average loss is {:1.5f}, including mse {:1.5f}, mae {:1.5f} and elbo {:1.5f}.-----------")
        logger.info(template.format(epoch, np.sum(loss_epoch.cpu().detach().numpy())/sample, np.sum(loss_mse.cpu().detach().numpy())/sample, np.sum(loss_mae.cpu().detach().numpy())/sample, np.sum(loss_elbo.cpu().detach().numpy())/(sample*1e4)))

        state = {
            'state_dict': model.state_dict()
        }
        torch.save(state, save_path)
        print('Model saved successfully:', save_path)
