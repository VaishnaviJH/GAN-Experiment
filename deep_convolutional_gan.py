
import argparse
import numpy as np
import os
import cv2

#Dependent library declarations
from gan_modules import GANModules
from image_generator import ImageGenerator

#keras libary declarations
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam


class DCGAN():
    def __init__(self, pre_load_weights, model_path, dataset_path):
        # configurations Input shape
        self.model_path = model_path
        self.img_height = 256
        self.img_width = 256
        self.channels = 3
        self.img_dimensions = (self.img_height, self.img_width, self.channels)
        self.input_latent_dimension = 1000

        # Configuration of image generator
        self.dataset_path = dataset_path
        self.image_generator = ImageGenerator(dataset_path=self.dataset_path,GAN_type = "DCGAN", img_dimensions = (self.img_height, self.img_width) )

        #model configurations
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator and discriminator model
        self.GAN_modules = GANModules(self.img_dimensions, self.channels, self.input_latent_dimension)
        self.generator = self.GAN_modules.dc_generator()
        self.discriminator = self.GAN_modules.dc_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z_input = Input(shape=(self.input_latent_dimension,))
        generated_img = self.generator(z_input)

        # The discriminator takes generated images as input and determines validity
        self.discriminator.trainable = False
        valid = self.discriminator(generated_img)

        # Trains the generator to fool the discriminator
        self.combined = Model(z_input, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        if (pre_load_weights):
            self.combined.load_weights(self.model_path)


    def train(self, epochs, batch_size=128, save_interval=50):

        #Adversarial loss ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for idx_batch, (imgs) in enumerate(self.image_generator.generate_batch_images(batch_size)):
                # generate noise and batch of images
                noise = np.random.normal(0, 1, (batch_size, self.input_latent_dimension))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid)

                # print the epoch and batch progress
                print ("Epoch of %d/%d and  Batch of %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f] " % (epoch, epochs, idx_batch, self.image_generator.n_batches, d_loss[0], 100*d_loss[1], g_loss))

                # save image at image_generation_period interval
                if idx_batch % save_interval == 0:
                    self.save_output_images(epoch, idx_batch)
            #Save the weights of the model
            self.combined.save_weights("DCModel.h5")

    def save_output_images(self, epoch, batch_i):
        os.makedirs('DCimages/%d' % (epoch), exist_ok=True)

        # test the GAN on batch of images
        ground_truth = self.image_generator.generate_img_data(batch_size=1, is_testing=True)
        noise = np.random.normal(0, 1, (1, self.input_latent_dimension))
        generated_image = self.generator.predict(noise)
        ground_truth = np.array(((0.5 * ground_truth[0] + 0.5) * 255)).astype('uint8').reshape(self.img_height, self.img_width, self.channels)
        generated_image = np.array(((0.5 * generated_image[0] + 0.5) * 255)).astype('uint8').reshape(self.img_height, self.img_width, self.channels)

        # save the input output files
        cv2.imwrite(os.path.join('DCimages/%d' % (epoch), str(batch_i) + '_GroundTruth.png'), ground_truth)
        cv2.imwrite(os.path.join('DCimages/%d' % (epoch), str(batch_i) + '_generatedimage.png'), generated_image)




if __name__ == '__main__':
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument("--loadweights", default=False, help="Load weight model.h5 (default True)")
    argParser.add_argument("--modelpath", default="DCModel.h5", help="Load model Path")
    argParser.add_argument("--dataset_path", default="cityscapes", help="Input dataset Path")
    args = argParser.parse_args()

    if(args.loadweights == "True" or args.loadweights == "true"):
        pre_load_weights = True
    else:
        pre_load_weights = False
    model_path = args.modelpath
    dataset_path = args.dataset_path

    # Process and train the GAN model
    dcgan = DCGAN(pre_load_weights, model_path, dataset_path)
    dcgan.train(epochs=400, batch_size=1, save_interval=50)
