from train_predict import *


if __name__ == '__main__':
    data_dir = "input/data/*.*"
    lr_shape = (64, 64, 3)    #Low Resolution image dimension
    hr_shape = (256, 256, 3)  #High Resolution image dimension
    mode = 'train'


    if mode == 'train':
        epochs = 3000
        batch_size = 2
        n_epoch  = 100     #Save results during training after every n_epochs

        train(data_dir, epochs , batch_size, lr_shape, hr_shape, n_epoch)

    if mode == 'predict':
        samples = 10        #Amount of images to predict

        predict(data_dir, samples, lr_shape, hr_shape )
