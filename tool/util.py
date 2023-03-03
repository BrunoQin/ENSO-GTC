import os
import torch
import cmaps
import netCDF4 as nc
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


def avg_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    N, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, out_h, out_w))

    for n in range(N):
        for i in np.arange(out_h):
            for j in np.arange(out_w):
                pool_z[n, i, j] = np.mean(padding_z[n,
                                                    strides[0] * i:strides[0] * i + pooling[0],
                                                    strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z

def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    N, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, out_h, out_w))

    for n in range(N):
        for i in np.arange(out_h):
            for j in np.arange(out_w):
                pool_z[n, i, j] = np.max(padding_z[n,
                                                   strides[0] * i:strides[0] * i + pooling[0],
                                                   strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z


def get_climatology(filename, scope=[1440, 1800]):
    climatology = None
    if os.path.exists('./file/climatology.npz'):
        climatology = np.load('./file/climatology.npz')['climatology']
    else:
        sst = np.array(nc.Dataset(filename, mode='r').variables['sst'])[range(scope[0], scope[1])]
        sst[sst == -1e+30] = 0
        sst[sst == -1000] = 0
        climatology = []
        for i in range(12):
            climatology.append(np.mean(sst[i::12], axis=0))
        climatology = np.array(climatology)
        np.savez('./file/climatology.npz', climatology=climatology)
    return climatology


def down_remote_sensing(filename, save_path):
    sst = np.load(filename)['sst']
    print(sst[0])
    sst[np.isnan(sst)] = 0
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    print(sst[0])
    sst = avg_pooling_forward(sst, (2, 2))
    sst = avg_pooling_forward(sst, (2, 2))
    print(sst.shape)
    np.savez(save_path, sst=sst)

def preprocess_raw_data(filename, save_path):
    sst = np.array(nc.Dataset(filename, mode='r').variables['sst'])
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 180*360))), (-1, 180, 360))
    np.savez(save_path, sst=sst)

def get_length_and_scaler(filename):
    sst = np.array(nc.Dataset(filename, mode='r').variables['sst'])
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 180*360))), (-1, 180, 360))
    return sst.shape[0], scaler

def preprocess_remote_sensing_data(filename, save_path):
    sst = np.load(filename)['sst']
    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 180*360))), (-1, 180, 360))
    np.savez(save_path, sst=sst)

def read_data(filename):
    return torch.Tensor(np.load(filename)['sst'])

def read_lons_and_lats(filename):
    ncfile = nc.Dataset(filename, mode='r')
    lons = np.array(ncfile.variables['longitude'][:])
    lats = np.array(ncfile.variables['latitude'][:])
    return lons, lats

def get_patch(data, m, n):
    h, w = data.shape
    data = data.view(m, int(h / m),
                     n, int(w / n))
    data = data.permute(0, 2, 1, 3).contiguous().view(-1, int(h / m), int(w / n))
    return data

def filter_sea_and_land(data, m, n, ratio=0.6):
    _, h, w = data.shape
    loc = np.count_nonzero(data, axis=(1, 2))
    loc = np.where(loc > h * w * ratio, 1, 0)
    loc = np.reshape(loc, (m, n))
    return loc.squeeze()

def get_physics_graph(loc, sequence_length=1):
    shape = loc.shape
    a = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    is_land = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            neighbor = []
            if (i - 1) >= 0:
                neighbor.append(((i - 1), j))
            if (i + 1) < shape[0]:
                neighbor.append(((i + 1), j))
            if (j - 1) >= 0:
                neighbor.append((i, (j - 1)))
            elif (j - 1) == -1:
                neighbor.append((i, shape[1] - 1))
            if (j + 1) < shape[1]:
                neighbor.append((i, (j + 1)))
            elif (j + 1) == shape[1]:
                neighbor.append((i, 0))

            if (i - 1) >= 0 and (j - 1) >= 0:
                neighbor.append(((i - 1), (j - 1)))
            if (i - 1) >= 0 and (j + 1) < shape[1]:
                neighbor.append(((i - 1), (j + 1)))
            if (i + 1) < shape[0] and (j - 1) >= 0:
                neighbor.append(((i + 1), (j - 1)))
            if (i + 1) < shape[0] and (j + 1) < shape[1]:
                neighbor.append(((i + 1), (j + 1)))

            for k in range(len(neighbor)):
                a[i * shape[1] + j, neighbor[k][0] * shape[1] + neighbor[k][1]] = 1
            if loc[i, j] == 0:
                is_land.append(i * shape[1] + j)
    a = np.delete(a, is_land, axis=0)
    a = np.delete(a, is_land, axis=1)
    assert np.sum(a - a.T) == 0

    if sequence_length == 1:
        adj = a
    else:
        a_eye = np.eye(a.shape[0])
        a_zero = np.zeros(a.shape)
        adj = []
        for i in range(sequence_length):
            row = []
            for j in range(sequence_length):
                if j == i:
                    row.append(a)
                elif j == i + 1:
                    row.append(a_eye)
                else:
                    row.append(a_zero)
            adj.append(row)
    adj = np.block(adj)
    deg = np.diag(np.sum(adj, axis=1))
    assert np.all(np.around(np.linalg.eigvals(deg - adj), 2) >= 0)
    return adj, deg

def construct_train_temporal_seqence(data, input_sequence_length, lead_time):
    input = torch.stack([data[i:i+input_sequence_length, ...] for i in range(data.shape[0] - input_sequence_length - lead_time + 1)])
    output = torch.stack([data[i+input_sequence_length+lead_time-1, ...] for i in range(data.shape[0] - input_sequence_length - lead_time + 1)])
    return input, output

def construct_forecast_temporal_seqence(data, input_sequence_length, lead_time):
    input = torch.stack([data[i:i+input_sequence_length, ...] for i in range(data.shape[0] - input_sequence_length - lead_time + 1, data.shape[0] - input_sequence_length + 1)])
    return input

def restore_pattern(data, loc):
    patch_land = np.zeros(data[0].shape)
    patch_land[patch_land == 0] = np.nan
    p = 0
    pattern = []
    for i in range(loc.shape[0]):
        row = []
        for j in range(loc.shape[1]):
            if loc[i, j] == 0:
                row.append(patch_land)
            else:
                row.append(data[p])
                p += 1
        pattern.append(row)
    return np.block(pattern)

def get_st_patch(data, loc, m, n):
    return torch.stack([np.delete(get_patch(data[i], m, n), np.where(loc.flatten() == 0), axis=0) for i in range(data.shape[0])])

def load_input_and_output(filename):
    data = np.load(filename, allow_pickle=True)
    return data['input'], data['output']

def plot_helper(data, lons, lats, save=True, filename='pred.png'):
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    # lons = lons - 180
    data = np.concatenate((data[:, 180:360], data[:, 0:180]), axis=1)
    plt.contourf(lons, lats, data, 60, transform=ccrs.PlateCarree(central_longitude=180), cmap=cmaps.CBR_wet_r)
    ax.coastlines()
    # plt.show()
    if save:
        plt.savefig(filename, bbox_inches='tight')
        plt.cla(); plt.clf(); plt.close()

def print_model_parameters(model, only_num=False):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
