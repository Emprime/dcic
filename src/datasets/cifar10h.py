from src.datasets.common.dataset_skeleton import DatasetSkeleton


class CIFAR10H(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'CIFAR10H',32,['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],[1953,1981,2067,1969,2030], 3, [1003,998,999,995,981,1004,1005,1015,999,1001],[0.4942142800245083, 0.4851313890165443, 0.4504090927542883], [0.24665251509498376, 0.24289226346005494, 0.2615923780220242], root_directory)
