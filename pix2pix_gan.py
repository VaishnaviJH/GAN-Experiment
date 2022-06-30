import argparse
import cv2
from gan_modules import GANModules
from image_generator import ImageGenerator
import numpy as np
import os

from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam



class Pix2PixGAN():
    def __init__(self, pre_load_weights, model_path, dataset_path):
        # Input shape
        self.model_path = model_path
        self.img_height = 256
        self.img_width = 256
        self.channels = 3
        self.img_dimensions = (self.img_height, self.img_width, self.channels)

        # Configure data loader
        self.dataset_path = dataset_path
        self.image_generator = ImageGenerator(dataset_path = self.dataset_path, GAN_type = "P2P", img_dimensions = (self.img_height, self.img_width) )
        patch = int(self.img_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.GAN_modules = GANModules(self.img_dimensions, self.channels)

        # Build the generator and discriminator model
        self.generator = self.GAN_modules.p2p_generator()
        self.discriminator = self.GAN_modules.p2p_discriminator()
        self.discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

        #Define the model
        groundtruth_img = Input(shape=self.img_dimensions)
        conditional_img = Input(shape=self.img_dimensions)

        # By giving the conditional image generate fake image of generator
        fake_img_of_generator = self.generator(conditional_img)

        self.discriminator.trainable = False

        # Discriminators determines whether the groundtruth image and conditional image is valid
        valid = self.discriminator([groundtruth_img, conditional_img])

        self.combined = Model(inputs=[groundtruth_img, conditional_img], outputs=[valid, fake_img_of_generator])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        if(pre_load_weights):
            self.combined = load_model(self.model_path)

    def train(self, epochs, batch_size=1, image_generation_period=50):

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (groundtruth_img, conditional_img) in enumerate(self.image_generator.generate_batch_images(batch_size)):
                fake_img_of_generator = self.generator.predict(conditional_img)
                real_discriminator_loss = self.discriminator.train_on_batch([groundtruth_img, conditional_img], valid)
                fake_discriminator_loss = self.discriminator.train_on_batch([fake_img_of_generator, conditional_img], fake)
                discriminator_loss = 0.5 * np.add(real_discriminator_loss, fake_discriminator_loss)
                generator_loss = self.combined.train_on_batch([groundtruth_img, conditional_img], [valid, groundtruth_img])

                #save image at image_generation_period interval
                if batch_i % image_generation_period == 0:
                    self.save_output_images(epoch, batch_i)
                print("Epoch of %d/%d  Batch of %d/%d" % (epoch, epochs, batch_i, self.image_generator.n_batches))
            self.combined.save('P2PModel.h5')

    def save_output_images(self, epoch, batch_i):
        os.makedirs('output_images/%d' % (epoch), exist_ok=True)

        groundtruth_img, conditional_img = self.image_generator.generate_img_data(batch_size=1, is_testing=True)
        fake_img_of_generator = self.generator.predict(conditional_img)

        groundtruth_img = np.array(((0.5 * groundtruth_img[0] + 0.5) * 255)).astype('uint8').reshape(self.img_height, self.img_width, self.channels)
        conditional_img = np.array(((0.5 * conditional_img[0] + 0.5) * 255)).astype('uint8').reshape(self.img_height, self.img_width, self.channels)
        fake_img_of_generator = np.array(((0.5 * fake_img_of_generator[0] + 0.5) * 255)).astype('uint8').reshape(self.img_height, self.img_width, self.channels)

        cv2.imwrite(os.path.join('output_images/%d' % (epoch), str(batch_i) + '_GroundTruth.png'), groundtruth_img)
        cv2.imwrite(os.path.join('output_images/%d' % (epoch), str(batch_i) + '_Conditional.png'), conditional_img)
        cv2.imwrite(os.path.join('output_images/%d' % (epoch), str(batch_i) + '_OutputOfGenerator.png'), fake_img_of_generator)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument("--loadweights", default = True, help = "Load weight model.h5 (default true)")
    argParser.add_argument("--modelpath", default = "model.h5", help = "Load model Path")
    argParser.add_argument("--dataset_path", default="cityscapes", help="Input dataset Path")
    args = argParser.parse_args()

    pre_load_weights = args.loadweights
    model_path = args.modelpath
    dataset_path = args.dataset_path
    p2pgan = Pix2PixGAN(pre_load_weights, model_path, dataset_path)
    p2pgan.train(epochs=200, batch_size=5, image_generation_period=100)
