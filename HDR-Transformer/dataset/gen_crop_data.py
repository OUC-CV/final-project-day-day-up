import os
import glob
import shutil
import argparse
import cv2
import random
import numpy as np

def get_croped_data_per_scene(scene_dir, patch_size=128, stride=64):
    exposure_file_path = os.path.join(scene_dir, 'exposure.txt')
    ldr_file_path = sorted(glob.glob(os.path.join(scene_dir, '*.tif')))
    label_path = os.path.join(scene_dir, 'HDRImg.hdr') 
    ldr_0 = cv2.imread(ldr_file_path[0], cv2.IMREAD_UNCHANGED)
    ldr_1 = cv2.imread(ldr_file_path[1], cv2.IMREAD_UNCHANGED)
    ldr_2 = cv2.imread(ldr_file_path[2], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    crop_data = []
    h, w, _ = label.shape # 1000x1500 for Kalantari17's dataset and 1500x1125 for ICCP19's dataset
    for x in range(w):
        for y in range(h):
            if x * stride + patch_size <= w and y * stride + patch_size <= h:
                crop_ldr_0 = ldr_0[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_ldr_1 = ldr_1[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_ldr_2 = ldr_2[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_label = label[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_sample = {
                    'ldr_0': crop_ldr_0, 
                    'ldr_1': crop_ldr_1, 
                    'ldr_2': crop_ldr_2, 
                    'label': crop_label,
                    'exposure_file_path': exposure_file_path
                    }
                crop_data.append(crop_sample)
    print("{} samples of scene {}.".format(len(crop_data), scene_dir))
    return crop_data

def rotate_sample(data_sample, mode=0):
    if mode == 0:
        flag = cv2.ROTATE_90_CLOCKWISE
    elif mode == 1:
        flag = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotate_ldr_0 = cv2.rotate(data_sample['ldr_0'], flag)
    rotate_ldr_1 = cv2.rotate(data_sample['ldr_1'], flag)
    rotate_ldr_2 = cv2.rotate(data_sample['ldr_2'], flag)
    rotate_label = cv2.rotate(data_sample['label'], flag)
    return {
        'ldr_0': rotate_ldr_0, 
        'ldr_1': rotate_ldr_1, 
        'ldr_2': rotate_ldr_2, 
        'label': rotate_label,
        'exposure_file_path': data_sample['exposure_file_path']
        }

def flip_sample(data_sample, mode=0):
    # mode: 0 for vertical flip and 1 for horizontal flip
    flip_ldr_0 = cv2.flip(data_sample['ldr_0'], mode)
    flip_ldr_1 = cv2.flip(data_sample['ldr_1'], mode)
    flip_ldr_2 = cv2.flip(data_sample['ldr_2'], mode)
    flip_label = cv2.flip(data_sample['label'], mode)
    return {
        'ldr_0': flip_ldr_0, 
        'ldr_1': flip_ldr_1, 
        'ldr_2': flip_ldr_2, 
        'label': flip_label,
        'exposure_file_path': data_sample['exposure_file_path']
        }

def adjust_brightness_contrast(data_sample, brightness=0, contrast=1.0):
    adjust_ldr_0 = cv2.convertScaleAbs(data_sample['ldr_0'], alpha=contrast, beta=brightness)
    adjust_ldr_1 = cv2.convertScaleAbs(data_sample['ldr_1'], alpha=contrast, beta=brightness)
    adjust_ldr_2 = cv2.convertScaleAbs(data_sample['ldr_2'], alpha=contrast, beta=brightness)
    adjust_label = cv2.convertScaleAbs(data_sample['label'], alpha=contrast, beta=brightness)
    return {
        'ldr_0': adjust_ldr_0, 
        'ldr_1': adjust_ldr_1, 
        'ldr_2': adjust_ldr_2, 
        'label': adjust_label,
        'exposure_file_path': data_sample['exposure_file_path']
    }


def flip_diagonal(data_sample):
    # 获取图像的高度和宽度
    height, width = data_sample['ldr_0'].shape[:2]

    # 创建一个沿对角线翻转的变换矩阵
    transform_matrix = np.float32([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    # 对所有图像应用对角线翻转
    flipped_ldr_0 = cv2.warpAffine(data_sample['ldr_0'], transform_matrix, (width, height))
    flipped_ldr_1 = cv2.warpAffine(data_sample['ldr_1'], transform_matrix, (width, height))
    flipped_ldr_2 = cv2.warpAffine(data_sample['ldr_2'], transform_matrix, (width, height))
    flipped_label = cv2.warpAffine(data_sample['label'], transform_matrix, (width, height))

    return {
        'ldr_0': flipped_ldr_0, 
        'ldr_1': flipped_ldr_1, 
        'ldr_2': flipped_ldr_2, 
        'label': flipped_label,
        'exposure_file_path': data_sample['exposure_file_path']
    }


def save_sample(data_sample, save_root, id):
    save_path = os.path.join(save_root, id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copyfile(data_sample['exposure_file_path'], os.path.join(save_path, 'exposure.txt'))
    cv2.imwrite(os.path.join(save_path, '0.tif'), data_sample['ldr_0'])
    cv2.imwrite(os.path.join(save_path, '1.tif'), data_sample['ldr_1'])
    cv2.imwrite(os.path.join(save_path, '2.tif'), data_sample['ldr_2'])
    cv2.imwrite(os.path.join(save_path, 'label.hdr'), data_sample['label'])

def main():
    parser = argparse.ArgumentParser(description='Prepare cropped data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default='D:\code\Vision\datas')
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--aug", action='store_true', default=False)

    args = parser.parse_args()

    full_size_training_data_path = os.path.join(args.data_root, 'Training')
    cropped_training_data_path = os.path.join(args.data_root, 'sig17_training_crop{}_stride{}'.format(str(args.patch_size), str(args.stride)))
    if not os.path.exists(cropped_training_data_path):
        os.makedirs(cropped_training_data_path)

    global counter
    counter = 0
    all_scenes = sorted(glob.glob(os.path.join(full_size_training_data_path, '*')))
    for scene in all_scenes:
        print("==>Process scene: {}".format(scene))
        scene_dir = os.path.join(args.data_root, scene)
        croped_data = get_croped_data_per_scene(scene_dir, patch_size=args.patch_size, stride=args.stride)
        for data in croped_data:
            save_sample(data, cropped_training_data_path, str(counter).zfill(6))
            counter += 1

            if args.aug:
                rotate_sample_0 = rotate_sample(data, 0)
                save_sample(rotate_sample_0, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

                # rotate_sample_1 = rotate_sample(data, 1)
                # save_sample(rotate_sample_1, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                flip_sample_0 = flip_sample(data, 0)
                save_sample(flip_sample_0, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

                # flip_sample_1 = flip_sample(data, 1)
                # save_sample(flip_sample_1, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

if __name__ == '__main__':
    main()
