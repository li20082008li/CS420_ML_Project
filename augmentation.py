import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img


def augmentation(path_aug='data_aug/'):
    if not os.path.exists(path_aug):
        os.mkdir(path_aug)
    if not os.path.exists(path_aug + 'tmp'):
        os.mkdir(path_aug + 'tmp')

    # read images
    train_path = 'dataset/train_img/'
    label_path = 'dataset/train_label/'
    train_imgs = glob.glob(train_path + '*.png')
    train_labels = glob.glob(label_path + '*.png')
    
    slices = len(train_imgs)
    print("train images: ", slices)
    if len(train_imgs) != len(train_labels) or len(train_imgs) == 0:
        print("trains can't match labels")
        return 0

    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(rotation_range=10, shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                 vertical_flip=True, fill_mode='constant', cval=0)
    # merge label and train
    print('Using real-time data augmentation.')
    # one by one augmentation
    for i in range(slices):
        img_t = load_img(train_path + str(i) + '.png', color_mode="grayscale")
        img_l = load_img(label_path + str(i) + '.png', color_mode="grayscale")
        x_t = img_to_array(img_t)
        x_l = img_to_array(img_l)
        s = np.shape(x_t)
        img = np.ndarray(shape=(s[0], s[1], 3), dtype=np.uint8)

        img[:, :, 0] = x_t[:, :, 0]
        img[:, :, 2] = x_l[:, :, 0]
        # here's a more 'manual' example
        img = img.reshape((1,) + img.shape)
        batches = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=path_aug + 'tmp', save_prefix=str(i),
                                  save_format='png'):
            batches += 1
            if batches >= 30:
                break
            # we need to break the loop by hand because
            # the generator loops indefinitely

    aug_imgs = glob.glob(path_aug + 'tmp/*.png')
    print("augment images: ", len(aug_imgs))
    dirs = ['train_img', 'train_label', 'val_img', 'val_label']
    for d in dirs:
        savedir = path_aug + d
        if not os.path.exists(savedir):
            os.mkdir(savedir)

    i = 0
    for imgname in aug_imgs:
        img = load_img(imgname)
        img = img_to_array(img)

        img_train = img[:, :, :1]
        img_label = img[:, :, 2:]
        img_train = array_to_img(img_train)
        img_label = array_to_img(img_label)
        if i < 700:
            img_train.save(path_aug + 'train_img/' + str(i) + '.png')
            img_label.save(path_aug + 'train_label/' + str(i) + '.png')
        else:
            img_train.save(path_aug + 'val_img/' + str(i) + '.png')
            img_label.save(path_aug + 'val_label/' + str(i) + '.png')
        i += 1


if __name__ == '__main__':
    augmentation()
