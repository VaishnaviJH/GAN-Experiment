from keras.layers import Activation, Input, Concatenate, ZeroPadding2D
from keras.layers import BatchNormalization, Dense, Reshape, Flatten, Dropout
from keras.layers.activation.leaky_relu import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model,Sequential

class GANModules():
    def __init__(self, img_dimensions, channels, input_latent_dimension = 100):
        # Number of filters in the first layer of G and D
        self.input_generator_filter = 64
        self.input_discriminator_filter = 64
        self.img_dimensions = img_dimensions
        self.channels = channels
        self.input_latent_dimension = input_latent_dimension

    def p2p_generator(self):
        def conv2d(input, filters, boolbn=True, filter_size=4):
            d_i = Conv2D(filters, kernel_size=filter_size, strides=2, padding='same')(input)
            d_i = LeakyReLU(alpha=0.2)(d_i)
            if boolbn:
                d_i = BatchNormalization(momentum=0.8)(d_i)
            return d_i

        def deconv2d(input, skip_input, filters, filter_size=4, dropout_rate=0):
            u_i = UpSampling2D(size=2)(input)
            u_i = Conv2D(filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(u_i)
            if dropout_rate:
                u_i = Dropout(dropout_rate)(u_i)
            u_i = BatchNormalization(momentum=0.8)(u_i)
            u_i = Concatenate()([u_i, skip_input])
            return u_i

        # Image input
        d0 = Input(shape=self.img_dimensions)

        # Downsampling
        d1 = conv2d(d0, self.input_generator_filter, boolbn=False)
        d2 = conv2d(d1, self.input_generator_filter*2)
        d3 = conv2d(d2, self.input_generator_filter*4)
        d4 = conv2d(d3, self.input_generator_filter*8)
        d5 = conv2d(d4, self.input_generator_filter*8)
        d6 = conv2d(d5, self.input_generator_filter*8)
        d7 = conv2d(d6, self.input_generator_filter*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.input_generator_filter*8)
        u2 = deconv2d(u1, d5, self.input_generator_filter*8)
        u3 = deconv2d(u2, d4, self.input_generator_filter*8)
        u4 = deconv2d(u3, d3, self.input_generator_filter*4)
        u5 = deconv2d(u4, d2, self.input_generator_filter*2)
        u6 = deconv2d(u5, d1, self.input_generator_filter)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def p2p_discriminator(self):

        def d_layer(layer_input, filters, boolbn=True, filter_size=4):
            """Discriminator layer"""
            d_i = Conv2D(filters, kernel_size=filter_size, strides=2, padding='same')(layer_input)
            d_i = LeakyReLU(alpha=0.2)(d_i)
            if boolbn:
                d_i = BatchNormalization(momentum=0.8)(d_i)
            return d_i

        GroundTruthImg = Input(shape=self.img_dimensions)
        ConditionalImg = Input(shape=self.img_dimensions)

        # Concatenate image and conditioning image by channels to produce input
        concatenate_imgs = Concatenate(axis=-1)([GroundTruthImg, ConditionalImg])

        d1 = d_layer(concatenate_imgs, self.input_discriminator_filter, boolbn=False)
        d2 = d_layer(d1, self.input_discriminator_filter*2)
        d3 = d_layer(d2, self.input_discriminator_filter*4)
        d4 = d_layer(d3, self.input_discriminator_filter*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([GroundTruthImg, ConditionalImg], validity)

    def dc_generator(self):

        model = Sequential()

        model.add(Dense(128 * 64 * 64, activation="relu", input_dim=self.input_latent_dimension))
        model.add(Reshape((64, 64, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.input_latent_dimension,))
        img = model(noise)

        return Model(noise, img)

    def dc_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_dimensions, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_dimensions)
        validity = model(img)

        return Model(img, validity)