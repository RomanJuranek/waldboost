import logging
import numpy as np
import waldboost as wb
from waldboost import utils

logging.basicConfig(level=logging.DEBUG)

channel_opts = { 
    "shrink": 2, 
    "n_per_oct": 4, 
    "smooth": 0, 
    "target_dtype": np.int32, 
    "channels": [ wb.channels.grad_hist_4 ] 
}

shape = (16,16,4)

model = wb.Model(shape, channel_opts)
learner = wb.Learner(alpha=0.2, max_depth=2)
pool = wb.SamplePool(1000,20000)

from waldboost.utils import fake_data_generator
training_images = fake_data_generator()

img,_ = next(training_images)

cb = [utils.ShowImageCallback(img)]

wb.train(model, training_images, learner=learner, pool=pool, length=16, callbacks=cb)

print(learner.__dict__)

wb.save(model, "x.pb")
learner.save("x.learner")
del model, learner, pool


print("-"*80)
model = wb.load("x.pb")
learner = wb.Learner.load("x.learner")
pool = wb.SamplePool(1000, 1000)

print(learner.true_positive_rate, learner.false_positive_rate)

wb.train(model, training_images, learner=learner, pool=pool, length=100, callbacks=cb)
