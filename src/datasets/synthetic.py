from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Synthetic(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Synthetic',224,['bc', 'be', 'gc', 'ge', 'rc', 're' ],[3000, 3000, 3000, 3000, 3000], 0, [2320,2635,2590,2540,2490,2425], [0.15772031678343634, 0.1689875409045194, 0.16716267022525277], [0.34661801691528576, 0.3570608578380146, 0.35501198420808544], root_directory)


