# wikiart-neural-trasnfer
## Author
Meng Chen
## Model
Abstract art has always been an intriguing topic for art lovers and researchers. In this first solution, I used a Deep Convolutional Generative Adversarial Network (DCGAN) model for generating abstract art.

Our DCGAN model consists of a generator and a discriminator. The generator takes a random noise vector as input and generates a 2D image as output. The discriminator is a binary classifier that takes an image as input and outputs a probability indicating whether the input is real or fake. Both the generator and discriminator are composed of several convolutional and transposed convolutional layers.

The generator consists of four transposed convolutional layers followed by a Tanh activation function. Each transposed convolutional layer increases the spatial resolution of the input by a factor of two. The discriminator consists of four convolutional layers followed by a sigmoid activation function. Each convolutional layer decreases the spatial resolution of the input by a factor of two. The loss function used in our model is the binary cross-entropy loss. We used the Adam optimization algorithm with a learning rate of 0.0002 and a momentum of 0.5.

## Evaluation
In our model, we used the binary cross-entropy loss, which measures the difference between the predicted output and the true output. The loss function of the generator is defined as:
-{(y\log(p) + (1 - y)\log(1 - p))}
where $z$ is the random noise vector, $G(z)$ is the output of the generator, and $D(\cdot)$ is the output of the discriminator. The generator aims to minimize this loss function to generate images that can fool the discriminator.

The loss function of the discriminator is defined as:


where $x$ is a real image, and $D(x)$ is the output of the discriminator for the real image. The discriminator aims to maximize this loss function to correctly classify real and fake images.

During training, the generator and discriminator are trained iteratively. The generator generates fake images, and the discriminator classifies the real and fake images. The gradients of the loss functions with respect to the parameters of the generator and discriminator are computed, and the parameters are updated accordingly.

We can evaluate the performance of our model by monitoring the loss values during training. Ideally, we want to see a decrease in the loss values over time, which indicates that the generator and discriminator are improving.
