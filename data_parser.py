import numpy as np
import torch
from sklearn.model_selection import train_test_split
from os import walk

# sample_name : (porosity(%), permeabilit[mD])
core_poro_perm_dict = {
    'Parker_2d25um_binary.raw' : (14.77, 10),
    'Kirby_2d25um_binary.raw' : (19.95, 62), 
    'BanderaBrown_2d25um_binary.raw' : (24.11, 63),
    'BSG_2d25um_binary.raw' : (19.07, 80),
    'BUG_2d25um_binary.raw' : (18.56, 86), 
    'Berea_2d25um_binary.raw' : (18.96, 121), 
    'CastleGate_2d25um_binary.raw' : (26.54, 269), 
    'BB_2d25um_binary.raw' : (24.02, 275), 
    'Leopard_2d25um_binary.raw' : (20.22, 327), 
    'Bentheimer_2d25um_binary.raw' : (22.64, 386)
}


def get_files_names(data_folder_path):
    filenames_cores = []
    for (dirpath, dirnames, filenames) in walk(data_folder_path):
        filenames_cores.extend(filenames)
        break

    return filenames_cores


def parse_files_into_array(data_folder_path, xyz_splits=[2,2,2], size=1):

    filenames_cores = get_files_names(data_folder_path)
    arrays_list = []
    poros_list = []
    permeabs_list = []
    ind = 0

    # Split each of the files and save to arrays_list
    for f_name_core in filenames_cores:
        
        if ind == size:
            break
            
        raw_file = np.fromfile(data_folder_path + f_name_core, dtype=np.uint8)
        sample = raw_file.reshape(1000,1000,1000)
        
        # x split
        x_split_array = np.array(np.split(sample, xyz_splits[0], axis=0))

        # y split
        y_split_array = np.array(np.split(x_split_array, xyz_splits[1], axis=2))

        # z split
        z_split_array = np.array(np.split(y_split_array, xyz_splits[2], axis=4))

        arrays_list.append(z_split_array.reshape(-1, z_split_array.shape[3], z_split_array.shape[4], z_split_array.shape[5]))

        total_samples_cnt = xyz_splits[0] * xyz_splits[1] * xyz_splits[2]
        poros_list.extend([core_poro_perm_dict[f_name_core][0]] * total_samples_cnt)
        permeabs_list.extend([core_poro_perm_dict[f_name_core][1]] * total_samples_cnt)
        
        ind += 1

    # save everything into one final array
    final_array = np.concatenate((arrays_list[0], arrays_list[1]), axis=0)
    del arrays_list[0]
    del arrays_list[0]

    for cur_array in arrays_list:
        final_array = np.concatenate((final_array, cur_array), axis=0)

    return final_array, np.array(poros_list), np.array(permeabs_list)


def get_torch_tensors(data_folder_path, test_size=0.2, xyz_splits=[2,2,2], size=1):
    X, y_poro, y_perm = parse_files_into_array(data_folder_path, xyz_splits, size=size)
    X_train, X_test, y_poro_train, y_poro_test, y_perm_train, y_perm_test = train_test_split(X, y_poro, y_perm, test_size=test_size)

    train_x = torch.from_numpy(X_train).float()
    train_y_poro = torch.from_numpy(y_poro_train).float()
    train_y_perm = torch.from_numpy(y_perm_train).float()
    test_x = torch.from_numpy(X_test).float()
    test_y_poro = torch.from_numpy(y_poro_test).float()
    test_y_perm = torch.from_numpy(y_perm_test).float()

    return train_x, train_y_poro, train_y_perm, test_x, test_y_poro, test_y_perm