#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from plot import *
from cluster import *


def plot(data, auto_select_dc=False):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dpcluster = DensityPeakCluster()
    distances, max_dis, min_dis, max_id, rho, dc = dpcluster.local_density(
        load_paperdata, data, auto_select_dc=auto_select_dc)
    delta, nneigh = min_distance(max_id, max_dis, distances, rho)
    plot_rho_delta(rho, delta)  # plot to choose the threshold


if __name__ == '__main__':
    plot('./data/data_jets/jets_lat8.forcluster', auto_select_dc=True)
