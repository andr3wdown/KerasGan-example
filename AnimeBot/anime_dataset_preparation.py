from PIL import Image
import numpy as np
from random import shuffle
import os
import cv2
import pickle

img_size = 64
path = 'datasets/anime/'


def get_files():
    return os.listdir(path)


def resize_and_pad(img, size):
    h, w = img.shape[:2]
    sh, sw = size

    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w/h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    col = [0, 0, 0, 0]

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img,
                                    pad_top,
                                    pad_bot,
                                    pad_left,
                                    pad_right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=col)
    for i in scaled_img:
        for j in range(len(i)):
            if i[j][3] == 0:
                i[j] = col

    return scaled_img


def create_random_image():
    arr = np.random.uniform(0, 255, (img_size, img_size, 3))
    arr = arr / 255
    return arr


def get_array_of_random_images(amount):
    images = []
    for i in range(amount):
        progress = np.round(i/amount * 100, 2)
        if progress % 1.0 == 0:
            print(progress, '% done with random')
        image = create_random_image()
        images.append(image)
    return images


def get_image_data(transparent=False, save_images=False):
    files = get_files()
    images = []
    for file in files:
        progress = np.round(files.index(file)/len(files) * 100, 2)
        if progress % 1.0 == 0:
            print(progress, '% done with images')
        image = cv2.imread(f'{path}{file}') if not transparent else cv2.imread(f'{path}{file}', cv2.IMREAD_UNCHANGED)
        if transparent:
            image = resize_and_pad(image, (64, 64))
        images.append(image)
    if not save_images:
        return images
    else:
        for i in range(len(images)):
            cv2.imwrite(files[i], images[i])


def create_classification_dataset():
    dataset = []
    anime_images = get_image_data()
    random_images = get_array_of_random_images(len(anime_images))

    for image in anime_images:
        dataset.append((image, [1]))
    for image in random_images:
        dataset.append((image, [0]))

    shuffle(dataset)

    X = []
    y = []
    for image, label in dataset[3000:]:
        X.append(image)
        y.append(label)
    pickle.dump(np.array(X), open('train_X.p', 'wb'))
    pickle.dump(np.array(y), open('train_y.p', 'wb'))
    print(f'files train_X.p and train_y.p have been saved!')
    test_X = []
    test_y = []
    for image, label in dataset[:3000]:
        test_X.append(image)
        test_y.append(label)
    pickle.dump(np.array(test_X), open('test_X.p', 'wb'))
    pickle.dump(np.array(test_y), open('test_y.p', 'wb'))
    print(f'files test_X.p and test_y.p have been saved!')


def create_gan_dataset(filename, save=True, transparent=False):
    dataset = []
    anime_images = get_image_data(transparent=transparent)
    for image in anime_images:
        dataset.append(image)
    shuffle(dataset)
    if save:
        pickle.dump(np.array(dataset), open(f'{filename}.p', 'wb'))
    else:
        return np.array(dataset)


if __name__ == '__main__':
    #create_classification_dataset()
    create_gan_dataset('datasets/anime_faces', transparent=False)

