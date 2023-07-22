import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
import torch

# import wandb
# wandb.init(project="ts_snr", entity="borealisai")

# series = AirPassengersDataset().load()
# series.plot()

# # train, val = series.split_before(pd.Timestamp("19580101"))
# # train.plot(label="training")
# # val.plot(label="validation")

# # from darts.models import NaiveSeasonal

# # naive_model = NaiveSeasonal(K=1)
# # naive_model.fit(train)
# # naive_forecast = naive_model.predict(36)

# # series.plot(label="actual")
# # naive_forecast.plot(label="naive forecast (K=1)")

# # wandb.log({"Media": plt})

# # wandb.finish()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# a = torch.tensor([1, 2, 3, 4, 5], device=device)
# print(a)

# import torch
# x = torch.tensor(2.0, requires_grad = True)
# print("x:", x)
# y = x**2
# y.backward()
# print(x.grad, y.is_leaf, x.is_leaf, x.grad.is_leaf)

import numpy as np

x = np.arange(1000000)
a = x.reshape(20, 50000)
print(a)

np.random.seed(0)
_laplace = np.random.laplace(0, 1, a.shape)
_normal = np.random.normal(0, 1, a.shape)
print(_laplace)

plt.hist(_laplace[0,:], bins=50)
plt.hist(_normal[0,:], bins=50, alpha=0.5)
plt.savefig("laplace.png")