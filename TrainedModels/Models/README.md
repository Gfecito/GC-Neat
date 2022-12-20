# This folder contains several trained Wolfram neural networks.

- DispNet follows a significant simplification the architecture proposed by Nikolaus et al: https://arxiv.org/pdf/1512.02134.pdf
- GCFull is the closest I could get to the original GCNet's architecture. You'll notice that I replaced transposed convolutions with standard convolutions, and avoided fiddling with image scaling. This is because I don't understand the effects that these 2 elements have, and therefore my analysis of their effect would've been uninteresting.
- GCFullReduced follows the same structure as GCFull, but significantly reduces the depth of the model.
- GCNetDemo was the architecture proposed during class.
- GCNetExtraConv is the same as the previous one, with 1 more convolutional layer.
- GCNetFullFeatures has all 2D convolution layers of the original paper for unary features, but without using residual layers.
- GCResFeaturesFull3D is the same as before, but with residual layers for 2D convolutions, as well as all the 3D convolutions (with transposed convolutions replaced with usual convolutions).
- GCResFeaturesReduced is the same as GCFullReduced without residuals on 3D convolutions.

Since GCNetDemo didn't have residuals on either 2D or 3D convolutions, I saw no point in making a new model that would also not have residuals on both, without other changes. 
