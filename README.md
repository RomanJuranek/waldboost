# WaldBoost detector

Object detection with WaldBoost detector for Python/Numpy. The detection
algortithm is similar to Aggregated Channel Features detector by Piotr Dollar.
The training algortithm is different - WaldBoost instead of Constant Soft Cascade.

The detector supports:
* Channel features - any channel type, shrinking, smoothing
* Decision tree or decision stump (dtree with depth 1) weak classifiers
* Output verification with CNN

NOTE: The implementation is not intended to be fast.
