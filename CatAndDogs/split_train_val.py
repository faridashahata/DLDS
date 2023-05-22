import os

import numpy as np
import pandas as pd

from typing import *


def load_data(path: str) -> pd.DataFrame:
    f = open(path, 'r')
    data: List[Tuple[str, int, int, int]] = []
    for line in f.readlines():
        split_line: List[str] = line.split(' ')

        data.append((split_line[0], int(split_line[1]), int(split_line[2]), int(split_line[3])))

    df = pd.DataFrame(data).rename(columns={0: 'file_name', 1: 'id', 2: 'species', 3: 'breed'})
    return df


def save_data(df: pd.DataFrame, path: str):
    df.to_pickle(path)


def label_to_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    breed = df['id']

    max_label: int = np.max(breed)
    one_hot_labels: List[np.array] = []

    for elem in breed:
        one_hot = np.zeros(max_label, dtype=int)
        one_hot[elem-1] = 1

        one_hot_labels.append(one_hot)

    df['id_onehot'] = one_hot_labels

    return df


def main():
    np.random.seed(42)

    training_ratio = 0.85

    path_to_train_val = "./data/annotations/trainval.txt"

    data = load_data(path_to_train_val)

    data = label_to_one_hot(data)

    indices = np.arange(len(data))
    np.random.shuffle(indices)

    n_train: int = int(np.floor(len(indices) * training_ratio))

    training_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_images = data.iloc[training_indices]
    val_images = data.iloc[val_indices]

    save_data(train_images, path='./data/annotations/train.pkl')
    save_data(val_images, path='./data/annotations/val.pkl')


if __name__ == '__main__':
    main()
