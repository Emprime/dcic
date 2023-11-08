from src.datasets.common.dataset_skeleton import DatasetSkeleton


class QualityMRI(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'QualityMRI',112,['0','1'],[60,64,62,62,62], 0, [108, 202], [0.14903500612533535, 0.14903500612533535, 0.14903500612533535], [0.1861035581616888, 0.1861035581616888, 0.1861035581616888], root_directory) #'1', '2', '3','4','5'

