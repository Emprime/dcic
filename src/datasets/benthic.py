from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Benthic(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Benthic',112,['coral', 'crustacean', 'cucumber', 'encrusting', 'other_fauna', 'sponge', 'star', 'worm'],[907, 955, 958, 990, 1057], 1, [506,471, 267, 1216, 856,566,465, 520],[0.34728872821176615, 0.40013687864974884, 0.4110478166769647], [0.1286915489786319, 0.13644626747739305, 0.14258506692263767],  root_directory)

