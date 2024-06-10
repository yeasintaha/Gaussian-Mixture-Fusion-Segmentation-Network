# Gaussian Mixture Fusion Segmentation Network
The Gaussian Mixture Fusion Segmentation Network is a deep learning architecture designed for image segmentation tasks, particularly suited for scenarios where integrating information from Gaussian mixture models (GMM) enhances segmentation accuracy.

This network comprises an encoder-decoder structure, where the encoder gradually downsamples the input image, and the decoder upsamples the encoded features to generate segmentation masks. Notably, it incorporates a novel fusion mechanism by concatenating feature maps from the original image, GMM segmentation maps, and intermediate convolutional layers at multiple stages of the network.

The fusion process allows the network to leverage both the raw pixel information from the input image and the refined segmentation cues provided by the GMM. By integrating these sources of information, the network learns to produce more accurate and robust segmentation results, especially in complex and ambiguous regions.

The architecture utilizes convolutional layers, batch normalization, activation functions (such as ReLU), dropout regularization, and pooling operations to learn hierarchical representations of the input data and facilitate effective feature extraction.

## Usage

Clone the repository:

```bash

```bash
git clone https://github.com/yeasintaha/Gaussian-Mixture-Fusion-Segmentation-Network.git

```
