#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from plot import *
from cluster import *


def plot(data, density_threshold, distance_threshold, auto_select_dc=False):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dpcluster = DensityPeakCluster()
    rho, delta, nneigh = dpcluster.cluster(
        load_paperdata, data, density_threshold, distance_threshold, auto_select_dc=auto_select_dc)
    logger.info(f'Centers as below, number of centers: {str(len(dpcluster.ccenter))}')
    for idx, center in dpcluster.ccenter.items():
        logger.info('%d %f %f' % (idx, rho[center], delta[center]))
    plot_rho_delta(rho, delta)   #plot to choose the threshold
    #plot_rhodelta_rho(rho,delta)
    plot_cluster(dpcluster)


if __name__ == '__main__':
    plot('./data/data_jets/lat8/jets_lat8.forcluster', density_threshold = 370, distance_threshold = 0.2, auto_select_dc=True)
