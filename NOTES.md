# 0.0.6.dev

## Package
* ~~Drop cv2 dependence (just numpy, skimage, scipy)~~ (still used somewhere, but not necessary anymore)

## Training
* ~~DTree weak classifier~~
* ~~Fern classifier~~ *CANCELED*
* ~~Agnostic to underlying feature type~~
* ~~Quantization~~
  * ~~quantized thresholds~~
  * ~~quantized response tables (hs)~~
* ~~Feature banks (fpga support)~~
* ~~fit_stage must return training history~~
* ~~Rejection scheduling in fit_stage~~
* ~~Degenerated solutions - training dstump on 0 samples~~
  * ~~1/ avoid if possible~~
  * ~~2/ solve when it happens~~
* ~~Check what happens with weights~~
* ~~Model saving and restoration (h5)~~ *Protobuf*

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
* ~~Stop updating pool at some point, e.g. P0 < 1e-7 or so~~
* ~~Sample pool without generator???~~
* ~~Sample pool should accept model in constructor (is mutable and can be used in update method without explicitly passing it)~~
* ~~Pool should not hold prior probs. should be in training module~~


# 0.1.0.dev

## Package
* Update Doc
* Input parameter checking - ValueError, TypeError where appropriate
* Example data and notebook
* Logging to screen and to file
* Applications
  * training - waldboost-train -i dataset --n0=1000 --alpha=0.2 -o test.pb
  * detection - waldboost-detect -m test.pb image.jpg image1.jpg ...

## Training
* Better API for Learner - methods object_prob, bg_prob, loss
* Resume learning
  * save/load learner to file
  * supply learer to training
* Constant soft cascade pipeline

## Model and channels
* Channel parametrs in proto
* Discard old messages from proto and simplify structure
* Correct symbol initialization when loading
* classifier as list of DTree, theta in its own array so it can be modified
* Optimizations
  * Numba and CUDA for one selected variant

## Data sampling
* Change image set during training
* ~~Either update bbx package to be more consistent or use bbox (or other) package~~
* ~~Better API for Pool~~
* Fix dt_boxes is None and gt_boxes is None cases

## Ground Truth
* ~~Improve bbGt reading - return~~

## Verification
* Verification - better log, validation set, show fp/fn rates
* bbox regression?

## FIXME
* SamplePool is slow when len(self.samples) gets large