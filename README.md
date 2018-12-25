# Google Quick Draw Challenge
TL;DR: SE-like models are currenly the best for image classification, however, CBAM-like models seem to be better (but there are no pretrained version of CBAM-ResNext models at the moment).

### Dataset
"Quick, Draw!" challenge offered a huge dataset with 49 millions of images. Images are represented in vector format, using stroke coordinates and timing information. This solution implements a traditional raster image classifier (think ResNet and others).

I reserved 1.5M samples for validation (every 30th sample). The rest of samples is split into blocks of 10k samples per class. For every epoch, I randomly select one block of each class, thus making the train set perfectly balanced.

A non-trivial part of this challenge was to understand how to pack vector image information into 3-channel RGB picture (see data_loader-* scripts).

### Metric
Metric is MAP@3 (mean average precision of top-3 predictions). The model is allowed to make 3 guesses for every image, however, a correct second guess would multiply its score by 0.5 and a correct third guess would multiply it by 0.33.

### Models
There are many different model here, ranging from simple ones (MobileNetV2, ResNet34) to quite advanced ones (SE-something):
* MobileNetV2
* SqueezeNet
* ResNet34
* ResNet50
* DenseNet121
* NasNet-A-Mobile
* DenseNet161
* SE-ResNext50
* DPN-68
* CBAM-ResNet50
* Inception-ResnetV2

### Image resolution
I used image resolutions ranging from 128x128 to 224x224 (default resolution for most ImageNet classification models). Later I found out that 128x128 is _enough for everyone_: it trains twice faster than 192x192, while quality is nearly the same.

### Learning
For most models, I use cosine annealing for periods of 32-64 epochs, increasing with multiplier of 1.2.

### Ensemble
A simple approach (blend2.py) is to blend predictions with some predefined weights, using mean or geometric mean. A better approach (unfortunately, not represented in this repo) is to train LightGBM model on top of all of these first-level models. It performed slightly better in our case (the score improved roughly by 0.001).
