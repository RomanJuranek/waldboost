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

from waldboost.utils import fake_data_generator
training_images = fake_data_generator()

wb.train(model, training_images, length=16, n_neg=200, n_pos=200)

wb.save(model, "x.pb")