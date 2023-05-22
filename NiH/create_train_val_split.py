import os

import numpy as np

from tqdm import tqdm

from typing import *


def load_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        lines = [line.rstrip() for line in file]
    return lines


def save_file(path_to_file: str, list_to_save: List[str]):
    with open(path_to_file, 'w') as file:
        for line in list_to_save:
            file.write(line + '\n')


def main():
    np.random.seed(42)

    training_ratio = 0.8

    train_val_list = np.stack(load_file('./data/train_val_list.txt'))
    test_images = np.stack(load_file('./data/test_list.txt'))

    indices = np.arange(len(train_val_list))
    np.random.shuffle(indices)

    n_train: int = int(np.floor(len(indices) * training_ratio))

    training_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_images = train_val_list[training_indices]
    val_images = train_val_list[val_indices]

    save_file('./data/train_list.txt', train_images)
    save_file('./data/val_list.txt', val_images)

    # MOVING ALL FILES
    base_dir: str = './data/images_resized'
    base_dir_train: str = './data/images_resized/train'
    base_dir_val: str = './data/images_resized/val'
    base_dir_test: str = './data/images_resized/test'

    if os.path.exists(base_dir_train):
        return
    
    os.mkdir(base_dir_train)
    os.mkdir(base_dir_val)
    os.mkdir(base_dir_test)
    
    for train_img_name in tqdm(train_images):
        os.rename(os.path.join(base_dir, train_img_name), os.path.join(base_dir_train, train_img_name))
    
    for val_img_name in tqdm(val_images):
        os.rename(os.path.join(base_dir, val_img_name), os.path.join(base_dir_val, val_img_name))

    for test_img_name in tqdm(test_images):
        os.rename(os.path.join(base_dir, test_img_name), os.path.join(base_dir_test, test_img_name))


if __name__ == '__main__':
    main()