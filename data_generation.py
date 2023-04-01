
import os
import sys
import cv2
import config
import numpy as np
import tensorflow as tf
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


label_names = config.classes.keys()
label_codes = config.classes.values()
code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}
name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}



def file_extension(x):
    if(x == 'images'):
        return 'jpg'
    elif(x == 'masks'):
        return 'png'


def aggregate_data(data_root_path, aggregate_data_folder):
    files = os.listdir(data_root_path)
    files.sort()
    #print(files)
    for folder in files:
        if(folder.split('.')[-1] != 'json'):
            folder_name_list = folder.split(' ')
            folder_name_joined = ''.join(folder_name_list)
            sub_folder_list = os.listdir(os.path.join(data_root_path, folder))
            for sub_folder in sub_folder_list:
                file_list = os.listdir(os.path.join(data_root_path, folder, sub_folder))
                file_list.sort()
                for i in range(len(file_list)):
                    source_file_path = os.path.join(data_root_path, folder, sub_folder, file_list[i])
                    new_filename = f'image_{folder_name_joined}_{i}.{file_extension(sub_folder)}'
                    destination_file_path = os.path.join(data_root_path, aggregate_data_folder, sub_folder, new_filename)
                    os.makedirs(os.path.join(data_root_path, aggregate_data_folder, sub_folder), exist_ok=True)
                    os.rename(source_file_path, destination_file_path)


def augment(width, height):
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
        A.OneOf([
            A.CLAHE (clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=1.0),
    ], p=1.0)

    return transform


def create_augmented_dataset(data_root_path, aggregate_data_folder, aug_multiplier):
    # Create folders for augmented data
    aug_folder_name = f'aug_data_{aug_multiplier}'
    images_dir = os.path.join(data_root_path, aggregate_data_folder, 'images')
    masks_dir = os.path.join(data_root_path, aggregate_data_folder, 'masks')
    aug_folder_images = os.path.join(data_root_path, aug_folder_name, 'images')
    aug_folder_masks = os.path.join(data_root_path, aug_folder_name, 'masks')
    os.makedirs(aug_folder_images, exist_ok=True)
    os.makedirs(aug_folder_masks, exist_ok=True)

    # Get filenames
    file_names = np.sort(os.listdir(images_dir))
    file_names = np.char.split(file_names, '.')
    filenames = np.array([])
    for i in range(len(file_names)):
        filenames = np.append(filenames, file_names[i][0])

    transform_1 = augment(512, 512)
    transform_2 = augment(480, 480)
    transform_3 = augment(512, 512)
    transform_4 = augment(800, 800)
    transform_5 = augment(1024, 1024)
    transform_6 = augment(800, 800)
    transform_7 = augment(1600, 1600)
    transform_8 = augment(1920, 1280)

    for i in range(aug_multiplier):
        for file in filenames:
            tile = file.split('_')[1]
            img = cv2.imread(os.path.join(images_dir, file + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(masks_dir, file + '.png'))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            if tile == 'Tile1':
                transformed = transform_1(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile2':
                transformed = transform_2(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile3':
                transformed = transform_3(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile4':
                transformed = transform_4(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile5':
                transformed = transform_5(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile6':
                transformed = transform_6(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile7':
                transformed = transform_7(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            elif tile =='Tile8':
                transformed = transform_8(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

            aug_image_path = f"{aug_folder_images}/aug_{str(i+1)+'_'+file+'.jpg'}"
            aug_mask_path = f"{aug_folder_masks}/aug_{str(i+1)+'_'+file+'.png'}"
            #print(aug_image_path, aug_mask_path)
            cv2.imwrite(aug_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            cv2.imwrite(aug_mask_path, cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))


def split_data(source, destination_train, destination_val, datatype, train_test_split = 0.8):
    counter = 0
    n = int(train_test_split*len(os.listdir(source)))
    aug_image_list = os.listdir(source)
    aug_image_list.sort()
    for filename in aug_image_list:
        source_path = os.path.join(source, filename)
        if(counter <= n):
            destination_path = os.path.join(destination_train, 'img', filename)
        else:
            destination_path = os.path.join(destination_val, 'img', filename)
        os.rename(source_path, destination_path)
        counter = counter + 1
    print(f'Augmented {datatype} split into train and val set!')


def split_dataset_into_train_and_val(data_root_path, train_test_split, aug_multiplier):
    # Create required folders
    aug_folder_name = f'aug_data_{aug_multiplier}'
    aug_folder_images = os.path.join(data_root_path, aug_folder_name, 'images')
    aug_folder_masks = os.path.join(data_root_path, aug_folder_name, 'masks')
    destination_aug_train_image = os.path.join(data_root_path, aug_folder_name, 'train', 'images')
    destination_aug_val_image = os.path.join(data_root_path, aug_folder_name, 'val', 'images')
    destination_aug_train_mask = os.path.join(data_root_path, aug_folder_name, 'train', 'masks')
    destination_aug_val_mask = os.path.join(data_root_path, aug_folder_name, 'val', 'masks')
    os.makedirs(os.path.join(destination_aug_train_image, 'img'), exist_ok=True)
    os.makedirs(os.path.join(destination_aug_val_image, 'img'), exist_ok=True)
    os.makedirs(os.path.join(destination_aug_train_mask, 'img'), exist_ok=True)
    os.makedirs(os.path.join(destination_aug_val_mask, 'img'), exist_ok=True)
    # Split images data
    split_data(aug_folder_images, destination_aug_train_image, destination_aug_val_image, 'images', train_test_split)
    # Split mask data
    split_data(aug_folder_masks, destination_aug_train_mask, destination_aug_val_mask, 'masks', train_test_split)


def rgb_to_onehot(rgb_image, colormap = id2code):
    # Function to one hot encode RGB mask labels
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap = id2code):
    # Function to decode encoded mask labels
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def AugmentDataGenerator(images_dir, masks_dir, seed = 1, batch_size = 8, target_size = (512, 512)):
    # Create neccessary variables
    label_names = config.classes.keys()
    label_codes = config.classes.values()
    code2id = {v:k for k,v in enumerate(label_codes)}
    id2code = {k:v for k,v in enumerate(label_codes)}
    name2id = {v:k for k,v in enumerate(label_names)}
    id2name = {k:v for k,v in enumerate(label_names)}

    frames_gen_args = dict(rescale=1./255)
    masks_gen_args = dict()
    frames_datagen = ImageDataGenerator(**frames_gen_args)
    masks_datagen = ImageDataGenerator(**masks_gen_args)

    image_generator = frames_datagen.flow_from_directory(images_dir, batch_size = batch_size, seed = seed, target_size = target_size, shuffle=False)
    mask_generator = masks_datagen.flow_from_directory(masks_dir, batch_size = batch_size, seed = seed, target_size = target_size, shuffle=False)

    while True:
        X1i = image_generator.next()
        X2i = mask_generator.next()

        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


  def calculate_number_of_samples(data_root_path, train_image_folder, val_image_folder, batch_size):
      num_train_samples = len(np.sort(os.listdir(os.path.join(data_root_path, train_image_folder, 'img'))))
      num_val_samples = len(np.sort(os.listdir(os.path.join(data_root_path, val_image_folder, 'img'))))
      steps_per_epoch = np.ceil(float(num_train_samples) / float(batch_size))
      #print('steps_per_epoch: ', steps_per_epoch)
      validation_steps = np.ceil(float(4 * num_val_samples) / float(batch_size))
      #print('validation_steps: ', validation_steps)
      return steps_per_epoch, validation_steps
