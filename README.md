# WaldBoost detector

Object detection with WaldBoost detector for Python/Numpy. The detection
algortithm is similar to Aggregated Channel Features detector by Piotr Dollar.
The training algortithm is different - WaldBoost instead of Constant Soft Cascade.

The detector supports:
* Channel features - any channel type, shrinking, smoothing
* Decision tree or decision stump (dtree with depth 1) weak classifiers
* Output verification with CNN

NOTE: The implementation is not intended to be fast.

**Acknowledgment** Development of this software was funded by ECSEL FitOptiVis project and V3C Center of Competence.

# Installation

Necessary requirements include:
* numpy
* scipy
* scikit-image
* scikit-learn
* opencv-python
* protobuf
* bbx

`waldboost.verification` module additionally require
* tensorflow-gpu (or pure tensorflow)

The package can be installed through `pip`

```sh
pip install waldboost-*.tgz
```

# Documentation

[Documentation](doc/api.md)

# Quick start

Following example show basic pipeline for training the detector.

Include the package and other required packages (e.g. dataset generators etc.)

```python
import waldboost as wb
# ...
```

Setup training parameters. Define how image channels are calculated and detector window size. For detailed info see `wb.channels.channel_pyramid`.

```python
channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 1,
    "target_dtype": np.float32,
    "channels": [ wb.grad_mag ]
}
shape = (12,12,1)
```

Initialize new model, sample pool (source of training data), and learner (training algorithm and state).

```python
model = wb.Model(shape, channel_opts)
pool = wb.Pool(shape, min_tp=1000, min_fp=1000)
learner = wb.Learner(alpha=0.2, wh=wb.DSKlearnDTree, max_depth=2))
```

Run the training. Each iteration updates training set from images produced by user-specified generator, and adds new stage to the model.

```python
for stage in range(len(model),T):
    pool.update(M, training_images)
    X0,H0 = P.gather_samples(0)
    X1,H1 = P.gather_samples(1)
    learner.fit_stage(model, X0, H0, X1, H1)
```

Finally model can be used for detection on new images, and saved to file.

```python
model.save("detector.pb")
image,*_ = next(training_images)
bbs, score = model.detect(image)
```

Function `wb.load_model` can load the model form file. Custom objects can be passed to the function in case that channel functions must be imported from other package.

```python
model = wb.Model.load("detector.pb")
```
