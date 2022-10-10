from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Turkey(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Turkey',192,['head_injury', 'not_injured', 'plumage_injury'],[1384,1761,1672,1431,1792], 0, [875, 1059, 6106],[0.42103789064934166, 0.5411304402006046, 0.42399450726330623], [0.213060781212991, 0.2503860141098106, 0.24951076340581185], root_directory)
