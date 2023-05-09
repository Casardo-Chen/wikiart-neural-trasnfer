# wikiart-neural-trasnfer
## Author
Meng Chen

## Introduction
This project aims to create new style abstract art based on neural-network-based models. I used three different models to generate abstract art: Deep Convolutional Generative Adversarial Network (DCGAN), Variational Autoencoder (VAE), and Neural Style Transfer. For Nueral Style Transfer, I also used a pre-trained VGG19 model to extract the style features of the style image. The result showed a combination of the style of the abstract art and the content of the content image(either a painting or a image).


## Data Set
Abstract Art Gallery: 
https://www.kaggle.com/datasets/bryanb/abstract-art-gallery

It contains two folders Abstract_gallery and Abstract_gallery_2. Abstract_gallery folder contains 2782 images of abstract art while Abstract_gallery_2 contains 90 images. The images are in the JPG format and have a resolution of 256x256 pixels. I used the Abstract_gallery folder for training.

Neural Transfer:
In Neural Transfer, I collected some art images and fun content images from the internet. The art images are from the following websites:


## GAN Model
Abstract art has always been an intriguing topic for art lovers and researchers. In this first solution, I used a Deep Convolutional Generative Adversarial Network (DCGAN) model for generating abstract art.

Our DCGAN model consists of a generator and a discriminator. \[2\]
![alt text](./res/gan_diagram.svg)

### Generator

The generator takes a random noise vector as input and generates a 2D image as output.

The architecture of the generator is shown in the following image:
![alt text](./res/dcgan_generator.png)

The generator consists of four transposed convolutional layers followed by a `Tanh` activation function. Each transposed convolutional layer increases the spatial resolution of the input by a factor of two.

I use Adam optimizer with a learning rate of 0.0002 and a momentum of 0.5. The loss function used in our model is the binary cross-entropy loss.

### Discriminator

The discriminator is a binary classifier that takes an image as input and outputs a probability indicating whether the input is real or fake. Both the generator and discriminator are composed of several convolutional and transposed convolutional layers.

| Layer Type | Output Shape | Kernal Size | Stride | Padding |Activation |
|:-----|:--------:|------:| ------:| ------:| ------:|
| Convolution | 32x3x64 | 4x4 | 2 | Same | LeakyReLU |
| Convolution | 16x16x128 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 16x16x128 | - | - | - | - |
| Convolution | 8x8x256 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 8x8x256 | - | - | - | - |
| Convolution | 4x4x512 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 4x4x512 | - | - | - | - |
| Convolution | 1x1x1 | 4x4 | 1 | Valid | Sigmoid |

The discriminator consists of four convolutional layers followed by a sigmoid activation function. Each convolutional layer decreases the spatial resolution of the input by a factor of two. The loss function used in our model is the binary cross-entropy loss. I also used the Adam optimization algorithm with a learning rate of 0.0002 and a momentum of 0.5 for optimizing the discriminator.

### Loss Function
The binary cross-entropy loss measures the difference between the predicted output and the true output. The loss function of the generator is defined as:

$L = -{(y\log(p) + (1 - y)\log(1 - p))}$

where $y$ is the true output, $p$ is the predicted output, and $y$ and $p$ are both scalars. The generator aims to minimize this loss function to generate images that can fool the discriminator.

The loss function of the discriminator is defined as:

$L_D = -\log(D(x)) - \log(1 - D(G(z)))$

where $x$ is a real image, $G(z)$ is the generated image, and $D(x)$ is the output of the discriminator for the real image. The discriminator aims to maximize this loss function to correctly classify real and fake images.

During training, the generator and discriminator are trained iteratively. The generator generates fake images, and the discriminator classifies the real and fake images. The gradients of the loss functions with respect to the parameters of the generator and discriminator are computed, and the parameters are updated accordingly.

## Evaluation
We can evaluate the performance of our model by monitoring the loss values during training. Ideally, we want to see a decrease in the loss values over time, which indicates that the generator and discriminator are improving.
![alt text](./res/loss_func.png)
The loss values of the first iteration are $Loss(D)$: 1.9443	$Loss(G)$: 6.4855 and the loss values of the last iteration are $Loss(D)$: 0.1673	$Loss(G)$: 4.2439. We can see that the loss values of the generator and discriminator are decreasing over time, which indicates that the generator and discriminator are improving. Howevver, the loss value occilates a lot from the 1000th iterations and does not decrease significantly after that. This is probably because the generator and discriminator are not able to learn from each other.

Another way to evaluate the performance of our model is to generate some images and visually inspect them. We can also use the discriminator to classify the generated images as real or fake. Ideally, we want to see that the discriminator classifies the generated images as real.

After 50 epochs, the images generated by our model look like this, which is not very ideal. There are some foggy noise in the images and some grid-like patterns.
![alt text](./res/50_epoch.jpeg)

After 100 epochs, the images generated by our model look like this, which is better than the previous one and have more artistic features. It has artistic strokes, reasonable color distribution, and some abstract patterns.
![alt text](./res/100_epoch.png)

I juxatposed the generated images with the real images and recruited 5 friends to rate the generated images. If the generated images are convincing, the rating should be close to 5. If the generated images are not convincing, the rating should be close to 1. The average rating of the generated images is 3.8, which is decent but not great. More training epochs and better hyperparameters may improve the performance of our model.

## Variational Autoencoder Model (VAE)
In this second solution, I used a Variational Autoencoder (VAE) model for generating abstract art. The architecture of the VAE model is shown in the following image:

![alt text](./res/vae.png)

VAE is a generative model that can generate new images by sampling from a latent space. It consists of an encoder and a decoder. The encoder takes an image as input and outputs a latent vector. The decoder takes the latent vector as input and generates an image. The VAE model is trained to minimize the reconstruction loss between the input image and the generated image. The latent vector is sampled from a normal distribution with zero mean and unit variance. The latent vector can be used to generate new images by sampling from the normal distribution.


### Encoder
The encoder takes an image as input and outputs a latent vector. The architecture of the encoder is shown in the following image:
| Layer Type | Output Shape | Kernal Size | Stride | Padding |Activation |
|:-----|:--------:|------:| ------:| ------:| ------:|
| Convolution | 32x3x64 | 4x4 | 2 | Same | LeakyReLU |
| Convolution | 16x16x128 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 16x16x128 | - | - | - | - |
| Convolution | 8x8x256 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 8x8x256 | - | - | - | - |
| Convolution | 4x4x512 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 4x4x512 | - | - | - | - |
| Convolution | 1x1x1 | 4x4 | 1 | Valid | Sigmoid |

### Decoder
| Layer Type | Output Shape | Kernal Size | Stride | Padding |Activation |
|:-----|:--------:|------:| ------:| ------:| ------:|
| Convolution | 32x3x64 | 4x4 | 2 | Same | LeakyReLU |
| Convolution | 16x16x128 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 16x16x128 | - | - | - | - |
| Convolution | 8x8x256 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 8x8x256 | - | - | - | - |
| Convolution | 4x4x512 | 4x4 | 2 | Same | LeakyReLU |
| Batch Normalization | 4x4x512 | - | - | - | - |
| Convolution | 1x1x1 | 4x4 | 1 | Valid | Sigmoid |

### Loss Function

### Evaluation


## Neural Transfer Model
Neural-Transfer, allows you to take an image and reproduce it with a new artistic style. The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image. \[4\]

To implement neural style transfer, we need to use a pre-trained VGG19 model. I chose a picture from the abstract art gallery dataset as the style image and a picture from the WikiArt dataset as the style image and pictures from Neural Style Transfer 
esize the images in the Abstract Art Gallery dataset to 64x64 pixels.

The architecture of the VGG19 model is shown in the following image: ![alt text](./res/nst-architecture.png)

### Architecture
VGG19 is a convolutional neural network that consists of 16 convolutional layers and 3 fully connected layers. The convolutional layers are used to extract features from the input image. The fully connected layers are used to classify the input image. The VGG19 model is trained on the ImageNet dataset, which contains 1.2 million images. I directly used the pre-trained VGG19 model as it has already learned to extract features from images.
The architecture of the VGG19 model is shown in the following image: ![alt text](./res/VGG-19-Architecture.png)

### Loss Function
The loss function of the neural style transfer model is the weighted sum of the content loss and the style loss. The content loss is the mean squared error between the feature maps of the input image and the feature maps of the content image. The style loss is the mean squared error between the Gram matrix of the feature maps of the input image and the Gram matrix of the feature maps of the style image.
### Evaluation
I tested 3 different style images and 3 different content images. The results are shown in the following images.
| Case | Input | Style | Output| 
|:-----|:--------:|------:| ------:|
| 1 | ![alt text](./res/nst_style_1.png) | ![alt text](./res/nst_input_1.png) | ![alt text](./res/nst_result_1.png) |


## Discussion and Future Work
One way to improve the performance of our model is to train the model for more epochs. However, the training time is very long, and I only have limited time to train the model. Another way to improve the performance of our model is to use a different model, such as a

I can also use a different dataset, such as the WikiArt dataset, to train our model. WikiArt is a large dataset of paintings from different artists. It contains 50,000 images of paintings in the JPG format and have a resolution of 256x256 pixels. I can use this dataset to train our model and generate paintings of other styles and conduct a neural style transfer.

## Reference
[1] https://arxiv.org/pdf/1511.06434v2.pdf

[2] https://developers.google.com/machine-learning/gan/gan_structure?hl=zh-cn

[3] https://learnopencv.com/variational-autoencoder-in-tensorflow/

[4] https://arxiv.org/abs/1508.06576