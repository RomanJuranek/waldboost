# Devplan

## Package
* Doc
* Assert checking
* Example data and notebook
* ~~Drop cv2 dependence (just numpy, skimage, scipy)~~ (still used somewhere, but not necessary anymore)
* setup
* Logging to screen and to file

## Training
* ~~DTree weak classifier~~
* Fern classifier
* ~~Agnostic to underlying feature type~~
* Quantization
  * quantized thresholds
  * quantized response tables (hs)
* Feature banks (fpga support)
* ~~fit_stage must return training history~~
* ~~Rejection scheduling in fit_stage~~
* Degenerated solutions - training dstump on 0 samples
  * ~~1/ avoid if possible~~
  * 2/ solve when it happens
* Check what happens with weights
* Model saving and restoration (h5)

## Channels
* ~~Configurable channels~~
* ~~Integer channels~~
* ~~More than one channel in image/sample~~

## Detector
* ~~Support for dtree and dstump~~

## Sampling and ground truth
* ~~bb distance must take sizes and aspect ratios into account~~
* ~~Random image adjustments~~
* ~~rewrite sample SamplePool~~
* ~~bbGt reader parameters~~
* ~~Ignore bounding boxes~~
* Stop updating pool at some point, e.g. P0 < 1e-7 or so
* ~~Sample pool without generator???~~
* ~~Sample pool should accept model in constructor (is mutable and can be used in update method without explicitly passing it)~~
* ~~Pool should not hold prior probs. should be in training module~~
* Change image set during training

## Verification
* Verification - better log, validation set, show fp/fn rates
* bbox regression?
