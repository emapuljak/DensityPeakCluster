#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# data reference : R. A. Fisher (1936). "The use of multiple measurements
# in taxonomic problems"

from distance_builder import *
from distance import *


if __name__ == '__main__':
    builder = DistanceBuilder()
    builder.load_points(r'../data/data_jets/jets_NA35Graviton_lat8.data')
    builder.build_distance_file_for_cluster(
        ManhattanDistance(), r'../data/data_jets/jets_NA35Graviton_lat8_l1dist.forcluster')
