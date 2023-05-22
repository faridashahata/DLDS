import os
from tqdm import tqdm
import cv2
from typing import *
import argparse

parser = argparse.ArgumentParser('Resize all images to 224x224 given a path')
parser.add_argument('original_path', type=str)
parser.add_argument('resized_path', type=str, default='./data/resized_images')


def resize_all_image_in_path(path: str, resized_path: str = './data/resized_images'):
    if not os.path.exists(path):
        print('Path does not exist')

    if not os.path.exists(resized_path):
        os.makedirs(resized_path)

    data_list = []

    files: List[str] = os.listdir(path)

    for file_name in tqdm(files):
        file_name: str

        if file_name.endswith('.png'):
            original_file_path: str = os.path.join(path, file_name)
            new_file_path: str = os.path.join(resized_path, file_name)

            image = cv2.imread(original_file_path)
            resized_img = cv2.resize(image, dsize=(224, 224))
            cv2.imwrite(new_file_path, resized_img)

            data_list.append([image, file_name])

    with open('data_list.txt', 'w') as f:
        for item in data_list:
            f.write(f"{item[0]};{item[1]}\n")


def main():
    args = parser.parse_args()

    original_path: str = args.original_path
    resized_path: str = args.resized_path

    resize_all_image_in_path(path=original_path,
                             resized_path=resized_path)


if __name__ == '__main__':
    main()
