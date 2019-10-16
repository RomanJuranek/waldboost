# WaldBoost detector

Object detection with WaldBoost detector for Python/Numpy. The detection
algortithm is similar to Aggregated Channel Features detector by Piotr Dollar.
The training algortithm is different - WaldBoost instead of Constant Soft Cascade.

The package supports
* Custom channel features - any channel type, shrinking, smoothing
* Decision tree weak classifiers
* FPGA-friendly training and inference pipeline

The purpose of this package is to provide reference implementation of detector training and inference of images for Python. It is not meant to be fast. We however did our best to speed up things using Numba while keeping the code as simple as possible.

**Acknowledgment** Development of this software was funded by TACR project and V3C Center of Competence (TE01020415) and ECSEL FitOptiVis (No 783162).

# Installation

Necessary requirements include:
* numpy
* numba
* scipy
* scikit-image
* scikit-learn
* opencv-python
* protobuf

Additionally, the package requires [*Tensorflow Object Detection API*](https://github.com/tensorflow/models) which implements manipulation with bounding boxes (which we found full-featured and comprehensive enough) and detector testing. This cannot be installed by pip and user need to install it manually. This is not good a solution and we will try to find another way how to install the API in an automated way in future releases.

The package can be installed through `pip`

```bash
pip install waldboost-*.tgz
```

# Quick start

Following example show basic pipeline for training the detector.

1. Include the package and other required packages (e.g. dataset generators etc.)

```python
import waldboost as wb
# ...
```

2. Setup training parameters. Define how image channels are calculated and detector window size. For detailed info see `wb.channels.channel_pyramid`.

```python
channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 1,
    "target_dtype": np.float32,
    "channels": [ wb.channels.grad_hist ]
}
shape = (12,12,4)
```

3. Initialize new model, sample pool (source of training data), and learner (training algorithm and state).

```python
model = wb.Model(shape, channel_opts)
pool = wb.Pool(shape, min_tp=1000, min_fp=1000)
learner = wb.Learner(alpha=0.2, max_depth=2))
```

4. Run the training. Each iteration updates training set from images produced by user-specified generator, and adds new stage to the model.

```python
for stage in range(len(model),T):
    pool.update(M, training_images)
    X0,H0 = P.gather_samples(0)
    X1,H1 = P.gather_samples(1)
    learner.fit_stage(model, X0, H0, X1, H1)
```

4. Finally model can be used for detection on new images, and saved to file.

```python
model.save("detector.pb")
image,*_ = next(training_images)
boxes = model.detect(image)
```

5. Function `wb.load_model` can load the model form file.

```python
model = wb.Model.load("detector.pb")
```