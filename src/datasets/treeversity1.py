from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Treeversity1(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Treeversity#1',224,['bark', 'bud', 'flower', 'fruit', 'leaf', 'whole_plant'],[1820, 1912, 1905, 1966, 1886], 0, [990, 1928, 3014, 1135, 1539, 883], [0.4453300184150276, 0.4511968333150217, 0.3712841036009189], [0.2340091739662385, 0.22787975691107748, 0.2605235872678352],root_directory)


