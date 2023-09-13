#! /usr/bin/env python
import h5py
import numpy as np
import math

if __name__ == '__main__':

    read_file = 'data/data_jets/lat4/latentrep_QCD_sig_test_lat4.h5'

    # read train data
    with h5py.File(read_file, "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]

        data_train = np.vstack([l1[:], l2[:]])
        np.random.shuffle(data_train)
    
    data_train = np.matrix(data_train[:20000])

    with open('data/data_jets/lat4/jets_test_lat4.data', 'w') as f:
        for sample in data_train:
            np.savetxt(f, sample, fmt='%.8f')