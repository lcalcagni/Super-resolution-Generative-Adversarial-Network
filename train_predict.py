import time

import tensorflow as tf
from keras.callbacks import TensorBoard

from networks import *
from inputs_outputs import *


##########################################

#TRAIN
#------------------------------------------

def train(data_dir, epochs , batch_size, lr_shape, hr_shape, n_epoch):

    #Output Paths
    logs_dir = "output/logs/"
    model_dir = "output/model"
    train_results_dir = "output/training_results"

    #Build and compile VGG network
    vgg=build_vgg()
    vgg.trainable=False
    compile_vgg(vgg)

    #Build and compile discriminator
    discriminator=build_discriminator()
    compile_discriminator(discriminator)

    # Build the generator network
    generator = build_generator()


    #-----------
    # Build and compile the combined network for training the generator
    # hr_input_shape = (256, 256, 3)     #High Resolution image dimension
    lr_input_shape = (64, 64, 3)       #Low Resolution image dimension

    # input_hr = Input(shape=hr_input_shape)
    input_lr = Input(shape=lr_input_shape)

    # Generate high-resolution images from low-resolution images
    gen_hr_imgs = generator(input_lr)

    # Extract feature maps of the generated images
    features_gen = vgg(gen_hr_imgs)

    # Make the discriminator network as non-trainable
    discriminator.trainable = False

    # Get the probability of generated high-resolution images
    probs = discriminator(gen_hr_imgs)

    #Create the model and compile
    combined_model = Model([input_lr], outputs=[probs, features_gen])
    combined_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=Adam(0.0001, 0.9))

    #-----------

    # Add Tensorboard
    tensorboard = TensorBoard(log_dir=logs_dir.format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)


    for epoch in range(epochs):
        print("Epoch:{}".format(epoch))

        #------
        #Train discriminator

        # Sample a batch of images
        hr_imgs, lr_imgs = load_imgs(data_dir=data_dir,
                                     batch_size=batch_size,
                                     lr_shape=lr_shape,
                                     hr_shape=hr_shape)

        # Normalize images
        hr_imgs = norm_imgs(hr_imgs)
        lr_imgs = norm_imgs(lr_imgs)


        # Generate high-resolution images from low-resolution images
        gen_hr_imgs = generator.predict(lr_imgs)

        # Generate batch of real and generated (fake) targets
        real_labels = np.ones((batch_size, 16, 16, 1))
        fake_labels = np.zeros((batch_size, 16, 16, 1))

        # Train the discriminator network on real and fake images
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_hr_imgs, fake_labels)

        # Calculate total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print("d_loss:", d_loss)


        #------
        #Train generator

        # Sample a batch of images
        hr_imgs, lr_imgs = load_imgs(data_dir=data_dir,
                                     batch_size=batch_size,
                                     lr_shape=lr_shape,
                                     hr_shape=hr_shape)

        # Normalize images
        hr_imgs = norm_imgs(hr_imgs)
        lr_imgs = norm_imgs(lr_imgs)

        # Extract feature maps for real high-resolution images
        img_features = vgg.predict(hr_imgs)

        valid_labels=np.ones((batch_size, 16, 16, 1))

        # Train the generator network
        g_loss = combined_model.train_on_batch(lr_imgs,
                                         [valid_labels, img_features])

        print("g_loss:", g_loss)

        # Write the losses to Tensorboard
        tensorboard_logs(tensorboard, 'g_loss', g_loss[0], epoch)
        tensorboard_logs(tensorboard, 'd_loss', d_loss[0], epoch)


        # Sample and save images after every N_epoch
        if epoch % n_epoch == 0:
            hr_imgs, lr_imgs = load_imgs(data_dir=data_dir,
                                         batch_size=batch_size,
                                         lr_shape=lr_shape,
                                         hr_shape=hr_shape)
            # Normalize images
            hr_imgs = norm_imgs(hr_imgs)
            lr_imgs = norm_imgs(lr_imgs)

            gen_imgs = generator.predict_on_batch(lr_imgs)

            # Denormalize images
            lr_imgs = denorm_imgs(lr_imgs)
            hr_imgs = denorm_imgs(hr_imgs)
            gen_imgs = denorm_imgs(gen_imgs)

            plot_save(lr_imgs, hr_imgs, gen_imgs, train_results_dir, epoch)

    save_models(generator, discriminator, model_dir)




##########################################

#PREDICT
#------------------------------------------
def predict(data_dir, samples, lr_shape, hr_shape):

    #Output Path
    results_dir="output/results"

    # Build networks
    discriminator = build_discriminator()
    compile_discriminator(discriminator)

    generator = build_generator()

    # Load models
    generator.load_weights("output/model/generator.h5")
    discriminator.load_weights("output/model/discriminator.h5")

    # Get samples number of random images
    hr_imgs, lr_imgs = load_imgs(data_dir=data_dir,
                                     batch_size=samples,
                                     lr_shape=lr_shape,
                                     hr_shape=hr_shape)

    # Normalize images
    lr_imgs = norm_imgs(lr_imgs)

    # Generate high-resolution images from low-resolution images
    gen_imgs = generator.predict_on_batch(lr_imgs)

    # Denormalize images
    lr_imgs = denorm_imgs(lr_imgs)
    gen_imgs = denorm_imgs(gen_imgs)


    plot_save(lr_imgs, hr_imgs, gen_imgs, results_dir)
    
