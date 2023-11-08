from absl import flags

from src.datasets.benthic import Benthic
from src.datasets.cifar10h import CIFAR10H
from src.datasets.mice import Mice
from src.datasets.pig import Pig
from src.datasets.plankton import Plankton
from src.datasets.quality_mri import QualityMRI
from src.datasets.synthetic import Synthetic
from src.datasets.treeversity1 import Treeversity1
from src.datasets.treeversity6 import Treeversity6
from src.datasets.turkey import Turkey
from src.datasets.verse import VerseBlended, VerseMask

FLAGS = flags.FLAGS

flags.DEFINE_string(name='data_root', default='/data/',
                    help='The path to the root directory for the benchmark')


def get_all_dataset_infos():
    return {
            'CIFAR10H':CIFAR10H(FLAGS.data_root),'MiceBone':Mice(FLAGS.data_root),'Turkey':Turkey(FLAGS.data_root),'Plankton':Plankton(FLAGS.data_root),
        'Benthic': Benthic(FLAGS.data_root), 'QualityMRI': QualityMRI(FLAGS.data_root), 'Synthetic': Synthetic(FLAGS.data_root),
        'Treeversity#1': Treeversity1(FLAGS.data_root), 'Treeversity#6': Treeversity6(FLAGS.data_root), 'Pig':Pig(FLAGS.data_root),
        'verse_blended-vps' : VerseBlended(FLAGS.data_root), 'verse_mask1-vps': VerseMask(FLAGS.data_root)
            }

def get_lite_dataset_infos():
    valid_datasets = ['Benthic','MiceBone','QualityMRI','Treeversity#6']
    return {k:v for k,v in get_all_dataset_infos().items() if k in valid_datasets}
