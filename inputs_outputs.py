import glob
import tensorflow as tf

import numpy as np
from scipy.misc import imread, imresize

#import matplotlib as mpl      #Uncomment if you are running in Mac OS
#mpl.use('TkAgg')              #Uncomment if you are running in Mac OS

import matplotlib.pyplot as plt


###########################################

#INPUTS
#------------------------------------------
#Load images from training dataset
def load_imgs(data_dir, batch_size, hr_shape, lr_shape):

    # List of all images inside the data directory
    all_imgs = glob.glob(data_dir)

    # Choose a random batch of images
    imgs_rndm = np.random.choice(all_imgs, size=batch_size)

    hr_imgs = []
    lr_imgs = []

    for img in imgs_rndm:

        image = imread(img, mode='RGB')
        image = image.astype(np.float32)

        # Resize the image
        image_hr = imresize(image, hr_shape)
        image_lr = imresize(image, lr_shape)

        # Good practices: Random horizontal flip
        if np.random.random() < 0.5:
            image_hr = np.fliplr(image_hr)
            image_lr = np.fliplr(image_lr)

        hr_imgs.append(image_hr)
        lr_imgs.append(image_lr)

    # Convert the lists to Numpy NDArrays
    return np.array(hr_imgs), np.array(lr_imgs)


###########################################

#OUTPUTS
#------------------------------------------
#Plot images
def plot_imgs(lr_image, original_image, generated_image, path):

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(lr_image)
    ax.axis("off")
    ax.set_title("Low Resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path)


#------------------------------------------
#Write logs for Tensorboard
def tensorboard_logs(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


#------------------------------------------
#Save figures
def plot_save(lr_imgs, hr_imgs, gen_imgs, path, epoch=-1):

    for index, img in enumerate(gen_imgs):
        if epoch != -1:
            plot_imgs(lr_imgs[index], hr_imgs[index], img, path+"/img_{}_{}".format(epoch, index))
        else:
            plot_imgs(lr_imgs[index], hr_imgs[index], img, path+"/gen_img_{}".format(index))




#------------------------------------------
#Save trained models
def save_models(generator, discriminator, path):
    generator.save_weights(path+"/generator.h5")
    discriminator.save_weights(path+"/discriminator.h5")


###########################################

#DATA MANIPULATION

#------------------------------------------
#Normalize images
def norm_imgs(imgs):
    r=127.5

    imgs = imgs / r - 1.
    return imgs

#------------------------------------------
#Denormalize images
def denorm_imgs(imgs):
    r=127.5

    imgs = (imgs + 1) * r
    imgs = imgs.astype(np.int)
    return imgs
