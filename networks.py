
from keras.applications import VGG19
from keras import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, LeakyReLU, PReLU, Add, Dense



###########################################

#GENERATOR
#------------------------------------------
#Build the generator network
def build_generator():

    #Residual Block
    def residual_block(input):

        res = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
        res = BatchNormalization(momentum=0.8)(res)
        res = PReLU(shared_axes=[1,2])(res)
        res = Conv2D(64, kernel_size=3, strides=1, padding='same')(res)
        res = BatchNormalization(momentum=0.8)(res)
        res = Add()([res, input])

        return res

    #------
    #Parameters
    input_shape = (64, 64, 3)     #Low Resolution image dimension
    res_blocks = 16               #Number of Residual Blocks


    #Input Layer for the Low Resolution image
    input_lr = Input(shape=input_shape)

    #Pre-Residual Block
    pre_res = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(input_lr)
    pre_res = PReLU(shared_axes=[1,2])(pre_res)

    #Residual Blocks
    res = residual_block(pre_res)
    for i in range(res_blocks-1):
        res = residual_block(res)

    #Post-Residual Block
    post_res = Conv2D(64, kernel_size=3, strides=1, padding='same')(res)
    post_res = BatchNormalization(momentum=0.8)(post_res)
    post_res = Add()([post_res, pre_res])

    #Upsampling Blocks
    upsamp_1 = UpSampling2D(size=2)(post_res)
    upsamp_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(upsamp_1)
    upsamp_1 = PReLU(shared_axes=[1,2])(upsamp_1)

    upsamp_2 = UpSampling2D(size=2)(upsamp_1)
    upsamp_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(upsamp_2)
    upsamp_2 = PReLU(shared_axes=[1,2])(upsamp_2)

    #Output Layer for the High Resolution image
    output_hr = Conv2D(filters=3, kernel_size=9, strides=1, padding='same', activation='tanh')(upsamp_2)

    #------
    #Create the model
    model = Model(inputs=[input_lr], outputs=[output_hr], name='generator')
    return model


###########################################

#DISCRIMINATOR
#------------------------------------------
#Build the discriminator network
def build_discriminator():

    #Convolution Block
    def conv_block(input, filter, strides=1, batchnorm=True):
        conv = Conv2D(filters=filter, kernel_size=3, strides=strides, padding='same')(input)
        conv = LeakyReLU(alpha=0.2)(conv)

        if batchnorm:
            conv = BatchNormalization(momentum=0.8)(conv)
        return conv

    #------
    #Parameters
    input_shape = (256, 256, 3)     #High Resolution image dimension
    filter=64                       #Initial filter

    #Input Layer for the High Resolution image
    input_hr = Input(shape=input_shape)

    #Convolution blocks
    conv = conv_block(input_hr, filter, batchnorm=False)
    conv = conv_block(conv, filter, strides=2)
    conv = conv_block(conv, filter*2)
    conv = conv_block(conv, filter*2, strides=2)
    conv = conv_block(conv, filter*4)
    conv = conv_block(conv, filter*4, strides=2)
    conv = conv_block(conv, filter*8)
    conv = conv_block(conv, filter*8, strides=2)

    #Dense layer
    dense_layer = Dense(filter*16)(conv)
    dense_layer = LeakyReLU(alpha=0.2)(conv)

    #Output classification layer
    output_class = Dense(1, activation='sigmoid')(dense_layer)

    #------
    #Create the model
    model = Model(inputs=[input_hr], outputs=[output_class], name='discriminator')
    return model


#------
#Compile the discriminator
def compile_discriminator(model):
    model.compile(
        loss='mse',
        optimizer=Adam(0.0001, 0.9),
        metrics=['accuracy']
    )


###########################################

#VGG
#------------------------------------------
#Build the VGG network
def build_vgg():

    input_shape = (256, 256, 3)     #High Resolution image dimension

    # Load a pre-trained VGG19 model trained on 'Imagenet' dataset
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]

    input_layer = Input(shape=input_shape)

    features = vgg(input_layer)

    #------
    #Create the model
    model = Model(inputs=[input_layer], outputs=[features])
    return model


#------
#Compile the VGG network
def compile_vgg(model):
    model.compile(
        loss='mse',
        optimizer=Adam(0.0001, 0.9),
        metrics=['accuracy']
    )
