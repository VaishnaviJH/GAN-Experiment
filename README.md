# GAN-Experiment

Keras implementation and comparison of the results on conditional pixel to pixel GAN and Convolutional GAN. 

# Usage
Example usage is shown below.<br />
**Note:** Downloaded the dataset from http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

### Pix2Pix GAN:
python pix2pixgan.py --loadweights True --modelpath "Path to saved model" --dataset_path "Path to the dataset" 

### Deep Convolutional GAN
python deep_convolutional_gan.py --loadweights True --modelpath "Path to saved model" --dataset_path "Path to the dataset"


# Results and Comparison

### Pixel to pixel GAN
**&emsp;&emsp;GroundTruth image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Input Conditional image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<br />**
![GroundTruth](https://user-images.githubusercontent.com/20317408/176627061-5c2e629c-678c-4527-a7ea-702bf2629bb0.png)
![Conditional](https://user-images.githubusercontent.com/20317408/176627106-34d14b3a-0571-4b99-b0ae-1a83625a7467.png)
<br /> **Output at Epoch 0 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Output at Epoch 30** <br />
![OutputOfGenerator](https://user-images.githubusercontent.com/20317408/176627128-42bd66ba-61d1-4fa7-9056-e7f2012101dc.png)
![OutputOfGenerator](https://user-images.githubusercontent.com/20317408/176627356-ccda5a3b-3d1d-485d-a0b2-85c8efc9aa3d.png)

### DC GAN
**&emsp;&emsp;GroundTruth image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Output at Epoch 0 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Output at Epoch 40<br />**
![GroundTruth](https://user-images.githubusercontent.com/20317408/176629428-85ee8999-358d-4640-8817-6c7e0fd7ee1e.png)
![outputat40](https://user-images.githubusercontent.com/20317408/176629614-7993e387-ffda-4ce2-b0c6-5f42e3aa2e8e.png)
![outputat0](https://user-images.githubusercontent.com/20317408/176629665-a589db76-6df8-4f9f-9a89-7bf29fd5bb90.png)

## Observations:
The conditional pixel to pixel GAN was able to predict the outline, texture and colour at the early stage of training since we are giving the map as a conditional input. But the Deep Convolutional GAN couldn't predict the scenary becauase it is complex for the model to understand the input structure and takes a lot of time to train. Hence Conditional pixel to pixel GAN works better than the DC GAN.

**Note: Beacause of the shortage of the time and resource I was able to train only 30 epochs.** 











