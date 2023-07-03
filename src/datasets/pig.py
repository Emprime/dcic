from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Pig(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Pig',96,['1_intact', '2_short', '3_fresh', '4_notVisible'],[1947,2129,2007,2039,2115], 1, [4221, 3032, 2183,801], [0.6089936435685781, 0.5285004122633025, 0.5061122982307459], [0.2615174318388238, 0.23601860429519766, 0.24243714836652566], root_directory)
