from src.datasets.common.dataset_skeleton import DatasetSkeleton


class VerseBlended(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'verse_blended-vps',128,['0','1','2','3'],[1354, 0, 0, 1252, 1155], 1, [3389,239,92,41],[0.5294289742328205, 0.5286790641472792, 0.5270972091572067], [0.1767669136303605, 0.17672722341018132, 0.17652723649567797],  root_directory)


class VerseMask(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'verse_mask1-vps',128,['0','1','2','3'],[1354, 0, 0, 1252, 1155], 1, [3389,239,92,41],[0.02650857741820896, 0.023992985376576015, 0.018703094913111954], [0.10702430438706818, 0.09852794071457519, 0.086788976268091],  root_directory)

