import numpy as np
from hmm import HMM
# create a random data set with 3 time series.
data = np.random.randn(3, 100, 2)
# create a model with 10 hidden states.
model = HMM(10, 2)
# fit the model
model.fit(data)
# print the trained model
print(model)
