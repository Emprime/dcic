from src.datasets.common.dataset_skeleton import DatasetSkeleton


class Plankton(DatasetSkeleton):
    def __init__(self, root_directory):
        DatasetSkeleton.__init__(self,'Plankton',96,['bubbles', 'collodaria_black', 'collodaria_globule', 'cop', 'det', 'no_fit', 'phyto_puff', 'phyto_tuft', 'pro_rhizaria_phaeodaria', 'shrimp'],[2444,2462,2468,2457,2449],1, [514, 853, 709, 1405, 1057, 3710, 572, 1602, 1115, 743], [0.9663359216202008, 0.9663359216202008, 0.9663359216202008], [0.10069729102981237, 0.10069729102981237, 0.10069729102981237], root_directory)
