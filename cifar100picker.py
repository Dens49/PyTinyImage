# source: https://github.com/iamgroot42/tinyImagesDatasetPython
# labels_cifar100.txt is also cloned from that repository
# indices_cifar100.txt is cloned from https://www.cs.toronto.edu/~kriz/cifar_indexes

import numpy as np
from tqdm import tqdm as progressbar
import tinyimage


def get_labels():
    labels = []
    with open("./labels_cifar100.txt", "r") as f:
        for label in f:
            labels.append(label.rstrip())
    return labels


def get_indices():
    indices = []
    with open("./indices_cifar100.txt", "r") as f:
        for index in f:
            indices.append(int(index.rstrip()))
    return indices


# this uses the cifar100 classes as keywords to search in tinyimages
# and checks how many images are found in tinyimages for each class/keyword that are
# part of cifar100 or not
def check_by_classes():
    keywords = get_labels()
    tinyimage.openTinyImage()
    cifar100_indices = get_indices()

    indexes_in_cifar100 = []
    indexes_not_in_cifar100 = []
    for keyword in keywords:
        print("class:", keyword)
        indexes = tinyimage.retrieveByTerm(keyword)
        keyword_indexes_in_cifar100 = []
        keyword_indexes_not_in_cifar100 = []
        for i in indexes:
            if i in cifar100_indices:
                keyword_indexes_in_cifar100.append(i)
            else:
                keyword_indexes_not_in_cifar100.append(i)
        indexes_in_cifar100.extend(keyword_indexes_in_cifar100)
        indexes_not_in_cifar100.extend(keyword_indexes_not_in_cifar100)
        print(
            f"for class {keyword} there are {len(keyword_indexes_in_cifar100)} images found that are in cifar100 and {len(keyword_indexes_not_in_cifar100)} that aren't in cifar100",
        )
        print("#####")

    tinyimage.closeTinyImage()

    print(
        f"For all cifar100 classes there are {len(indexes_in_cifar100)} images found that are in cifar100 and {len(indexes_not_in_cifar100)} that aren't in cifar100"
    )


def extract_from_indices_file():
    base_output_dir = (
        "/mnt/data/tiny_images/py-tiny-image-access/loaded_images/cifar100"
    )
    # the first 50'000 indices are for training
    train_dir = base_output_dir + "/train"
    # the last 10'000 indices are for testing
    test_dir = base_output_dir + "/test"

    tinyimage.openTinyImage()
    cifar100_indices = get_indices()
    for i, index in enumerate(progressbar(cifar100_indices)):
        if i < 50000:
            output_dir = train_dir
        else:
            output_dir = test_dir
        meta = tinyimage.getMetaData(index)
        tinyimage.sliceToImage(tinyimage.sliceToBin(index), output_dir + "/" + meta[1])
    tinyimage.closeTinyImage()


if __name__ == "__main__":
    check_by_classes()
    # extract_from_indices_file()
