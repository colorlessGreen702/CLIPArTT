from .oxford_pets import OxfordPets
from .ucf101 import UCF101
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA


dataset_list = {
                "oxford_pets": OxfordPets,
                "ucf101": UCF101,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "imagenet-a": ImageNetA,
                "imagenet-v": ImageNetV2,
                }


def build_dataset(dataset, root_path):
    return dataset_list[dataset](root_path)