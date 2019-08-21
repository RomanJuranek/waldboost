import numpy as np
import waldboost as wb


channel_opts = { 
    "shrink": 2, 
    "n_per_oct": 4, 
    "smooth": 0, 
    "target_dtype": np.int32, 
    "channels": [ wb.channels.grad_hist_4 ] 
}


shape = (16,16,4)

model = wb.Model(shape, channel_opts)
learner = wb.Learner(alpha=0.01)
pool = wb.SamplePool(10000,10000)

from waldboost.utils import fake_data_generator
training_images = fake_data_generator()

wb.train(model, training_images, learner=learner, length=4)

print(learner.__dict__)

wb.save(model, "x.pb")
learner.save("x.learner")
del model, learner, pool


print("-"*80)
model = wb.load("x.pb")
learner = wb.Learner.load("x.learner")
pool = wb.SamplePool(10000, 10000)

print(learner.true_positive_rate, learner.false_positive_rate)

wb.train(model, training_images, learner=learner, length=16)
