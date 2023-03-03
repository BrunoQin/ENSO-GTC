import torch
import netCDF4 as nc
import numpy as np

from tool.util import preprocess_raw_data, read_data, filter_sea_and_land, get_st_patch, construct_train_temporal_seqence, construct_forecast_temporal_seqence, get_patch
from tool.configs import args


if __name__ == '__main__':

    m = int(args['init_h'] / args['patch_size'])
    n = int(args['init_w'] / args['patch_size'])

    data = preprocess_raw_data(filename=args['nc_file'], save_path=args['raw_file'])
    data = read_data(filename=args['raw_file'])
    loc = filter_sea_and_land(get_patch(data[0], m=m, n=n), m=m, n=n, ratio=args['ratio'])
    st_patch = get_st_patch(data, loc, m=m, n=n)
    input, output = construct_train_temporal_seqence(st_patch, input_sequence_length=args['sequence_length'], lead_time=args['lead_time'])
    np.savez(args['train_npz_file'], input=input, output=output)

    input = construct_forecast_temporal_seqence(st_patch, input_sequence_length=args['sequence_length'], lead_time=args['lead_time'])
    np.savez(args['forecast_npz_file'], input=input, output=None)
    # print(input.shape)
    # print(output.shape)

    # pattern = restore_pattern(output[0], loc)
    # print(pattern.shape)
