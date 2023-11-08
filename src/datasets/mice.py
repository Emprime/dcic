from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Mice(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'MiceBone',112,["g",'nr','ug'],[1540,1513,1325,1319,1543], 0, [1068, 5103, 1069], [0.4443307598418903, 0.4443307598418903, 0.4443307598418903], [0.24633541064133654, 0.24633541064133654, 0.24633541064133654], root_directory)
