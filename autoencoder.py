
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal, ndimage, misc

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Dropout, GaussianNoise
from keras.layers import Activation
from keras.layers import Conv2D, Deconv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv3D, UpSampling3D, MaxPooling3D
from keras.models import Model
from keras import regularizers
from keras import backend as K_backend
from keras import objectives




class Autoencoder(object):
    """
    Base class for all-purpose autoencoder. VAE, CNN-AE, etc will be built off of this.

    Input -> Encoder -> Z Latent Vector -> Decoder -> Output
    """
    def __init__(self,
                 input_shape=(28, 28, 1),
                 latent_dim=2,  # Size of the encoded vector
                 batch_size=100, # size of minibatch
                 compile_decoder=False # create the decoder. Not necessary for every use case
                 ):
        self.model = None
        self.encoder = None
        self.decoder = None
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.compile_decoder = compile_decoder
        assert K_backend.image_dim_ordering() == 'tf', 'Cannot support Theano ordering! Use TF ordering! #tensorflowmasterrace'

        # input image dimensions
        self.input_shape = input_shape
        # self.data_shape = input_shape[1:] # Shape of a single sample
        if len(input_shape) == 4:
            self.img_rows, self.img_cols, self.img_stacks, self.img_chns = input_shape
        elif len(input_shape) == 3:
            self.img_rows, self.img_cols, self.img_chns = input_shape
        elif len(input_shape) == 2:
            self.img_rows, self.img_cols = input_shape
            self.img_chns = 1
        elif len(input_shape) == 1:
            self.img_rows = input_shape[0]  # todo: test this
        else:
            raise ValueError("Invalid input shape: {}".format(input_shape))

    def rollup_decoder(self, z, z_input, layers_list):
        """
        Takes a list of Keras layers and returns the decoder back-half and the standalone decoder model
        :param z: Layer corresponding to the latent space vector
        :param z_input: Layer corresponding to the decoder input
        :param layers_list: List of layers to roll up
        :return:
        """
        ae = AE_Dec()
        dc = AE_Dec()
        last_ae = z
        last_dc = z_input
        for i, layer in enumerate(layers_list):
            #             if i ==0:
            last_ae = layer(last_ae)
            if self.compile_decoder:
                last_dc = layer(last_dc)
        return last_ae, last_dc


class AE_Dec(object):
    """
    Dummy object for reasons I can't remember. This may be deprecated.
    """
    def __init__(self):
        pass


class VAE(Autoencoder):
    """
    Variational Autoencoder.
    """
    def __init__(self,
                 input_shape=(28, 28, 1),
                 latent_dim=2,  # Size of the encoded vector
                 batch_size=100,  # size of minibatch
                 epsilon_std=1.0, # This is the stddev for our normal-dist sampling of the latent vector
                 compile_decoder=False
                 ):
        super().__init__(input_shape=input_shape, latent_dim=latent_dim, batch_size=batch_size,
                         compile_decoder=compile_decoder)
        # Necessary to instantiate this as instance variables such that they can be passed to the loss function (internally), since loss functions are
        # all of the form lossfn(y_true, y_pred)
        self.epsilon_std = epsilon_std
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)



    def sampling(self, args):
        """
        This is what makes the variational technique happen.
        :param args:
        :return:
        """
        # Forging our latent vector from the reparameterized mean and std requires some sampling trickery
        # that admittedly I do not understand in the slightest at this point in time
        z_mean, z_log_var = args
        epsilon = K_backend.random_normal(shape=(self.batch_size, self.latent_dim),
                                          mean=0., stddev=self.epsilon_std)
        # We return z_mean + epsilon*sigma^2. Not sure why we use log var
        # Basically, create a random variable vector from the distribution
        # We are learning a distribution (mu, var) which represents the input
        return z_mean + K_backend.exp(z_log_var) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        """
        Custom loss function for VAE. Uses Kullback-Leibler divergence.

        Notes from fchollet: binary_crossentropy expects a shape (batch_size, dim) for x and x_decoded_mean,
        so we MUST flatten these!
        :param x:
        :param x_decoded_mean:
        :return:
        """

        x = K_backend.flatten(x)
        x_decoded_mean = K_backend.flatten(x_decoded_mean)
        shape_coef = np.product(self.input_shape)
        xent_loss = shape_coef * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K_backend.mean(
            1 + self.z_log_var - K_backend.square(self.z_mean) - K_backend.exp(self.z_log_var), axis=-1)
        # Kullback–Leibler divergence. so many questions about this one single line
        return xent_loss + kl_loss


class VAE_MNIST_0(VAE):
    """ Covolutional VAE for MNIST. Should work for other things, but untested """

    def __init__(self,
                 input_shape=(28, 28, 1),
                 latent_dim=2,  # Size of the encoded vector
                 n_classes=10,  # number of classes in dataset
                 batch_size=100,  # size of minibatch
                 n_stacks=3,  # Number of convolayers to stack, this boosts performance of the network dramatically
                 intermediate_dim=256,  # Size of the dense layer after convs
                 n_filters=64,  # Number of filters in the first layer
                 px_conv=3,  # Default convolution window size
                 dropout_p=0.1,  # Default dropout rate
                 epsilon_std=1.0,  # This is the stddev for our normal-dist sampling of the latent vector
                 compile_decoder=True,
                 ):

        # This is my original crossfire network, and it works. As such, it has apprentice marks all over
        # Reconstructing as-is before tinkering
        # Based heavily on https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py
        # and https://groups.google.com/forum/#!msg/keras-users/iBp3Ngxll3k/_GbY4nqNCQAJ

        super().__init__(input_shape=input_shape, latent_dim=latent_dim, batch_size=batch_size, epsilon_std=epsilon_std,
                         compile_decoder=compile_decoder)



        # Convolutional frontend filters as per typical convonets
        print(self.input_shape)
        x_in = Input(batch_shape=(batch_size,) + self.input_shape, name='main_input')
        conv_1 = Conv2D(n_filters, px_conv, px_conv, border_mode='same', activation='relu')(x_in)
        conv_2 = Conv2D(n_filters, px_conv, px_conv, border_mode='same', activation='relu',
                        subsample=(2, 2))(conv_1)
        stack = Conv2D(n_filters, px_conv, px_conv, border_mode='same', activation='relu',
                       name='stack_base')(conv_2)

        # I call this structure the "stack". By stacking convo layers w/ BN and dropout, the performance
        # of the network increases dramatically. For MNIST, I like n_stacks=3.
        # Presumably, the deepness allows for greater richness of filters to emerge
        for i in range(n_stacks):
            stack = BatchNormalization()(stack)
            stack = Dropout(dropout_p)(stack)
            stack = Conv2D(n_filters, px_conv, px_conv, border_mode='same', activation='relu',
                           name='stack_{}'.format(i), subsample=(1, 1))(stack)

        stack = BatchNormalization()(stack)
        conv_4 = Conv2D(n_filters, px_conv, px_conv, border_mode='same', activation='relu')(stack)

        # Densely connected layer after the filters
        flat = Flatten()(conv_4)
        hidden_1 = Dense(intermediate_dim, activation='relu', name='intermezzo')(flat)

        # This is the Variational Autoencoder reparameterization trick
        z_mean = Dense(latent_dim)(hidden_1)
        z_log_var = Dense(latent_dim)(hidden_1)

        # Make these instance vars so X-Ent can use them. Probably a better way out there
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        # Part 2 of the reparam trick is sample from the mean-vec and std-vec (log_var). To do this, we utilize a
        # custom layer via Lambda class to combine the mean and log_var outputs and a custom sampling function
        # 'z' is our latent vector
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='latent_z')([z_mean, z_log_var])

        # This marks the end of the encoding portion of the VAE

        # The 'classer' is a subnet after the latent vector, which will drive the distribution in order to
        # (hopefully) provide better generalization in classification
        # Note: in the original Crossfile I attach this layer to z_mean, rather than z, for reasons I cannot recall
        # I suspect this is because for classification, we do not care about the variance, just the mean of the vec
        # In this setup, we go straight to one-hot
        # Original uses normal init. Could try glorot or he_normal
        # todo: test behavior of attachment point of the classer, different inits
        classer_base = Dense(n_classes, init='normal', activation='softmax', name='classer_output')(self.z_mean)
        #         classer_base = Dense(n_classes, init='normal', activation='softmax', name='classer_output')(z)

        batch_size_dec = batch_size

        # On to Decoder. we instantiate these layers separately so as to reuse them later
        # e.g. for feeding in latent-space vectors, or (presumably) inspecting output
        decoder_hidden = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(n_filters * 14 * 14, activation='relu')


        output_shape = (batch_size_dec, 14, 14, n_filters)

        decoder_reshape = Reshape(output_shape[1:])  # FC's, I don't understand why this is here

        # FC uses Deconv, but another example uses UpSample layers. See Keras Api: Deconvolution2D
        decoder_deconv_1 = Deconv2D(n_filters, px_conv, px_conv, output_shape,
                                    border_mode='same', activation='relu')
        decoder_deconv_2 = Deconv2D(n_filters, px_conv, px_conv, output_shape,
                                    border_mode='same', activation='relu')

        # Some more reshaping, presumably I need to modify this in order to use different shapes
        output_shape = (batch_size_dec, 29, 29, n_filters)

        # more FC voodoo
        decoder_deconv_3_upsamp = Deconv2D(n_filters, 2, 2, output_shape, border_mode='valid', subsample=(2, 2),
                                           activation='relu')
        decoder_mean_squash = Conv2D(self.img_chns, 2, 2, border_mode='same', activation='sigmoid', name='main_output')

        # Now, piecemeal the encoder together. IDK why this is done this manner, and not functional like the
        # encoder half. presumably, this is so we can inspect the output at each point
        # hid_decoded = decoder_hidden(z)
        # up_decoded = decoder_upsample(hid_decoded)
        # reshape_decoded = decoder_reshape(up_decoded)
        # deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        # deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        # x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        # x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        hid_decoded = decoder_hidden
        up_decoded = decoder_upsample
        reshape_decoded = decoder_reshape
        deconv_1_decoded = decoder_deconv_1
        deconv_2_decoded = decoder_deconv_2
        x_decoded_relu = decoder_deconv_3_upsamp
        x_decoded_mean_squash = decoder_mean_squash

        layers_list = [decoder_hidden, decoder_upsample, decoder_reshape, decoder_deconv_1, decoder_deconv_2,
                       decoder_deconv_3_upsamp, decoder_mean_squash]

        decoder_input = Input(shape=(latent_dim,))

        # todo: better naming convention
        ae, dc = self.rollup_decoder(z, decoder_input, layers_list)

        if self.compile_decoder:
            # FC: build a digit generator that can sample from the learned distribution
            # todo: (un)roll this
            _hid_decoded = decoder_hidden(decoder_input)
            _up_decoded = decoder_upsample(_hid_decoded)
            _reshape_decoded = decoder_reshape(_up_decoded)
            _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
            _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
            _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
            _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)

        # Now we create the actual models. We also compile them automatically, this could be isolated later
        # Primary model - VAE
        self.model = Model(x_in, ae)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

        if False:

            # Crossfire network
            self.classifier = Model(x_in, classer_base)
            self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Ok, now comes the tricky part. See these references:
            # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
            # I believe the names have to match the layer names, but are otherwise arbitrary
            self.crossmodel = Model(input=x_in, output=[x_decoded_mean_squash, classer_base])
            self.crossmodel.compile(optimizer='rmsprop',
                                    loss={'main_output': self.vae_loss, 'classer_output': 'categorical_crossentropy'},
                                    loss_weights={'main_output': 1.0, 'classer_output': 5.0})

        # build a model to project inputs on the latent space
        self.encoder = Model(x_in, self.z_mean)
        if self.compile_decoder:
            # reconstruct the digit pictures from latent space
            self.decoder = Model(decoder_input, dc)

    def fit_crossmodel(self, x_dict, y_dict, batch_size=None, nb_epoch=10, verbose=1, callbacks=[],
                       validation_split=0.,
                       validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.crossmodel.fit(x_dict, y_dict, batch_size, nb_epoch, verbose, callbacks,
                                                validation_split,
                                                validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history


class FlatVAE(VAE):
    def __init__(self, input_shape=(365, 1),
                 latent_dim=2,  # Size of the encoded vector
                 n_classes=10,  # number of classes in dataset
                 batch_size=100,  # size of minibatch
                 n_stacks=3,  # Number of convolayers to stack, this boosts performance of the network dramatically
                 intermediate_dim=8,  # Size of the dense layer after convs
                 n_filters=4,  # Number of filters in the first layer
                 px_conv=3,  # Default convolution window size
                 dropout_p=0.1,  # Default dropout rate
                 epsilon_std=1.0,  # This is the stddev for our normal-dist sampling of the latent vector
                 ):
        super().__init__(input_shape=input_shape, latent_dim=latent_dim, batch_size=batch_size, epsilon_std=epsilon_std)

        # Dimensionality params
        d_x, d_w = input_shape  # vector len x vector chans
        d_x2 = d_x  # (d_x+1)//2 # after convolution border='same' and subsamp=2
        d_K = 8  # latent_dim*4 # penultimate latent layer dims

        x_in = Input(batch_shape=(batch_size,) + self.input_shape, name='main_input')
        bn_1 = BatchNormalization()(x_in)
        #         cnn_1 = Conv1D(n_filters, px_conv, activation='relu',
        #                        border_mode='same', subsample_length=2)(bn_1)
        d_1 = Dense(intermediate_dim, activation='relu', name='Enc_d_1')(bn_1)
        d_2 = Dense(d_K, activation='relu', name='Enc_d_2')(d_1)
        flat = Flatten()(d_2)
        hidden_1 = Dense(latent_dim * 2, activation='relu', name='intermezzo')(
            flat)  # CNN conversion layer, not really necessary here

        # One weird VAE trick
        z_mean = Dense(latent_dim, name='Z_mean')(hidden_1)
        z_log_var = Dense(latent_dim, name='Z_log_var')(hidden_1)
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        z = Lambda(self.sampling, output_shape=(latent_dim,), name='latent_z')([z_mean, z_log_var])
        #         z = Dense(latent_dim, name='latent_z_basic')(z_mean)

        # Create our prototypes for the decoder section. We won't functionally connect them yet
        dec_hidden = Dense(latent_dim * 2, activation='relu', name='Dec_intermezzo')
        dec_d_2 = Dense(d_x2, activation='relu', name='Dec_d_2')
        dec_reshape = Reshape((d_x2, d_K), name='Dec_Reshape')
        dec_d_1 = Dense(d_x * intermediate_dim, activation='relu', name='Dec_d_1')
        dec_reshape2 = Reshape((d_x * intermediate_dim,), name='Dec_Reshape2')
        dec_d_0 = Dense(d_x, activation='relu', name='Dec_d_0')
        dec_reshape3 = Reshape((d_x, 1), name='Dec_Reshape3')

        #         dec_cnn_1 = UpSampling2D(size=(2,1))
        dec_squash = Activation('sigmoid', name='main_output')

        # Complete the AE back half
        ae_dec_hidden = dec_hidden(z)
        ae_dec_d_2 = dec_d_2(ae_dec_hidden)
        #         ae_dec_reshape = dec_reshape(ae_dec_d_2)
        #         ae_dec_d_1 = dec_d_1(ae_dec_reshape)
        #         ae_dec_r_2 = dec_reshape2(ae_dec_d_1)
        ae_dec_d_0 = dec_d_0(ae_dec_d_2)
        ae_dec_r_3 = dec_reshape3(ae_dec_d_0)

        layers_list = [dec_hidden, dec_d_2, dec_d_0, dec_reshape3, dec_squash]

        #         ae_dec_d_1 = dec_d_1(ae_dec_d_2)
        #         ae_dec_cnn_1 = dec_cnn_1(ae_dec_d_1)
        x_dec_squash = dec_squash(ae_dec_r_3)
        #         x_dec_squash = dec_squash(ae_dec_cnn_1)

        # Complete the Decoder back half
        dc_decoder_input = Input(shape=(latent_dim,))
        #         dc_dec_hidden = dec_hidden(dc_decoder_input)
        # #         dc_dec_reshape = dec_reshape(dc_dec_hidden)
        #         dc_dec_d_2 = dec_d_2(dc_dec_hidden)
        #         dc_dec_d_1 = dec_d_1(dc_dec_d_2)
        #         x_dc_dec_squash = dec_squash(dc_dec_d_1)

        ae, dc = self.rollup_decoder(z, dc_decoder_input, layers_list)

        #         mn_dec_hidden = dec_hidden(z)
        #         mn_dec_d_2 = dec_d_2(ae_dec_hidden)
        #         mn_dec_d_0 = dec_d_0(mn_dec_hidden)


        # Assemble our models
        self.model = Model(x_in, ae)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.encoder = Model(x_in, self.z_mean)
        if self.compile_decoder:
            self.decoder = Model(dc_decoder_input, dc)

    pass

class SimpleAE3D(object):
    def __init__(self, input_shape=(48, 48, 48, 1), n_classes=10, n_filters=4):
        """
        5D tensor with shape: (samples, channels, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th' or
        5D tensor with shape: (samples, conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'.
        """
        #         input_img = Input(shape=(1, 28, 28, 28)) # th =(nChan, nFrames, xPix, yPix) or (nChan, z, x, y)
        input_img = Input(shape=input_shape)  # th =(nChan, nFrames, xPix, yPix) or (nChan, z, x, y)
        img_chans = 1

        x = Conv3D(n_filters, 3, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
        x = Conv3D(n_filters * 8, 3, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
        x = Conv3D(n_filters * 16, 3, 3, 3, activation='relu', border_mode='same')(x)
        self.z = MaxPooling3D((2, 2, 2), border_mode='same')(x)

        # at this point the representation is (4, 4, 4, 8) i.e. 512-dimensional

        flat = Flatten()(self.z)
        classer_base = Dense(n_classes, init='normal', activation='softmax', name='classer_output')(flat)

        x = Conv3D(n_filters * 16, 3, 3, 3, activation='relu', border_mode='same')(self.z)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(n_filters * 8, 3, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        #         x = Conv3D(16, 3, 3, 3, activation='relu', border_mode='valid')(x)
        x = UpSampling3D((2, 2, 2))(x)
        # smash the extra channels down
        x = Conv3D(img_chans, 3, 3, 3, activation='sigmoid', border_mode='same')(x)
        #         print(input_shape)
        #         x = Dense(input_shape)(x) # fudge to fix output shape
        self.decoded = x
        self.autoencoder = Model(input_img, self.decoded)

        self.classifier = Model(input_img, classer_base)
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

