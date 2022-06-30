# GAN-Experiment

Keras implementation and comparison of the results on conditional pixel to pixel GAN and Convolutional GAN. 

# Usage
Example usage is shown below.<br />
**Note:** Downloaded the dataset from http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

### Pix2Pix GAN:
python pix2pixgan.py --loadweights True --modelpath "Path to saves model" --dataset_path "Path to the dataset" 

### Deep Convolutional GAN
python deep_convolutional_gan.py --loadweights True --modelpath "Path to saves model" --dataset_path "Path to the dataset"
