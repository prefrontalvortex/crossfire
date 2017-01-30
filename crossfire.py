
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Deconv2D
from keras.models import Model
from keras import backend as K_backend
from keras import objectives
from keras.utils import np_utils


class Autoencoder2D(object):
    def __init__(self,
                 input_shape=(28, 28, 1),
                 latent_dim=2,  # Size of the encoded vector
                 batch_size=100): # size of minibatch
        self.model = None
        self.encoder = None
        self.generator = None
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim

class VariationalAutoencoder2D(Autoencoder2D):
    def __init__(self,
                 input_shape=(28, 28, 1),
                 latent_dim=2,  # Size of the encoded vector
                 batch_size=100,  # size of minibatch
                 epsilon_std=1.0):  # This is the stddev for our normal-dist sampling of the latent vector):
        super().__init__(input_shape=input_shape, latent_dim=latent_dim, batch_size=batch_size)
        self.epsilon_std = epsilon_std
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)

        # input image dimensions
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.img_rows, self.img_cols, self.img_chns = input_shape
        elif len(input_shape) == 2:
            self.img_rows, self.img_cols = input_shape
            self.img_chns = 1
        else:
            raise ValueError("Invalid input shape: {}".format(input_shape))


    def sampling(self, args):
        # Forging our latent vector from the reparameterized mean and std requires some sampling trickery
        # that admittedly I do not understand in the slightest at this point in time
        z_mean, z_log_var = args
        epsilon = K_backend.random_normal(shape=(self.batch_size, self.latent_dim),
                                          mean=0., std=self.epsilon_std)
        return z_mean + K_backend.exp(z_log_var) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        # Custom loss function for VAE. More VAE voodoo
        # FC: NOTE: binary_crossentropy expects a batch_size by dim
        # FC: for x and x_decoded_mean, so we MUST flatten these!
        x = K_backend.flatten(x)
        x_decoded_mean = K_backend.flatten(x_decoded_mean)
        xent_loss = self.img_rows * self.img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K_backend.mean(
            1 + self.z_log_var - K_backend.square(self.z_mean) - K_backend.exp(self.z_log_var), axis=-1)
        # Kullback–Leibler divergence. so many questions about this one single line
        return xent_loss + kl_loss



class Crossfire_MNIST_0(VariationalAutoencoder2D):
    """ Covolutional VAE with a "crossfire" component - a classifier is bolted onto the end of the encoder
    and loss function can be parameterized to function from classifier loss instead of autoencoder loss.
    Ideally, the model starts in pure autoencoder mode to learn features, then as loss flattens out, the
    network starts weighing classifier loss progressively greater """

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
                 ):

        # This is my original crossfire network, and it works. As such, it has apprentice marks all over
        # Reconstructing as-is before tinkering
        # Based heavily on https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py
        # and https://groups.google.com/forum/#!msg/keras-users/iBp3Ngxll3k/_GbY4nqNCQAJ

        super().__init__(input_shape=input_shape, latent_dim=latent_dim, batch_size=batch_size, epsilon_std=epsilon_std)



        # theano vs tensorflow ordering of size parameters
        if K_backend.image_dim_ordering() == 'th':
            self.original_img_size = (self.img_chns, self.img_rows, self.img_cols)
        else:
            self.original_img_size = (self.img_rows, self.img_cols, self.img_chns)


        # experimental
        def_conv = {'border_mode': 'same', 'activation': 'relu'}

        # Convolutional frontend filters as per typical convonets
        x_in = Input(batch_shape=(batch_size,) + self.original_img_size, name='main_input')
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

        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size_dec, n_filters, 14, 14)
        else:
            output_shape = (batch_size_dec, 14, 14, n_filters)

        decoder_reshape = Reshape(output_shape[1:])  # FC's, I don't understand why this is here

        # FC uses Deconv, but another example uses UpSample layers. See Keras Api: Deconvolution2D
        decoder_deconv_1 = Deconv2D(n_filters, px_conv, px_conv, output_shape,
                                    border_mode='same', activation='relu')
        decoder_deconv_2 = Deconv2D(n_filters, px_conv, px_conv, output_shape,
                                    border_mode='same', activation='relu')

        # Some more reshaping, presumably I need to modify this in order to use different shapes
        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size_dec, n_filters, 29, 29)
        else:
            output_shape = (batch_size_dec, 29, 29, n_filters)

        # more FC voodoo
        decoder_deconv_3_upsamp = Deconv2D(n_filters, 2, 2, output_shape, border_mode='valid', subsample=(2, 2),
                                           activation='relu')
        decoder_mean_squash = Conv2D(self.img_chns, 2, 2, border_mode='valid', activation='sigmoid', name='main_output')

        # Now, piecemeal the encoder together. IDK why this is done this manner, and not functional like the
        # encoder half. presumably, this is so we can inspect the output at each point
        hid_decoded = decoder_hidden(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        # FC: build a digit generator that can sample from the learned distribution
        # todo: (un)roll this
        decoder_input = Input(shape=(latent_dim,))
        _hid_decoded = decoder_hidden(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)

        # Now we create the actual models. We also compile them automatically, this could be isolated later
        # Primary model - VAE
        self.model = Model(x_in, x_decoded_mean_squash)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

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
        # reconstruct the digit pictures from latent space
        self.generator = Model(decoder_input, _x_decoded_mean_squash)



    def fit_crossmodel(self, x_dict, y_dict, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                       validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.crossmodel.fit(x_dict, y_dict, batch_size, nb_epoch, verbose, callbacks,
                                                validation_split,
                                                validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history