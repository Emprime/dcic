from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Treeversity6(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Treeversity#6',112,['bark', 'bud', 'flower', 'fruit', 'leaf', 'whole_plant'],[1892, 1972, 1972, 2032, 1958], 0, [1011, 1839, 3072, 1170, 1872, 862],[0.4439581940620345, 0.4509297096690951, 0.3691211738638277], [0.23407518616927706, 0.22764417468550843, 0.2600833107790479], root_directory)
