"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 


import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor


import os
from pathlib import Path
import sys
from typing import List, Union, Literal

import torch
import wandb

from neuralop.training.training_state import save_training_state
from neuralop.utils import compute_rank, compute_stable_rank, compute_explained_variance

from neuralop.training.callbacks import Callback



class BasicLoggerCallback2(Callback):
    """
    Callback that implements simple logging functionality
    expected when passing verbose to a Trainer
    """

    def __init__(self, wandb_kwargs=None):
        super().__init__()
        if wandb_kwargs:
            wandb.init(**wandb_kwargs)
        self.loss = []
        self.val_loss = []

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_train_start(self, **kwargs):
        self._update_state_dict(**kwargs)

        train_loader = self.state_dict["train_loader"]
        test_loaders = self.state_dict["test_loaders"]
        verbose = self.state_dict["verbose"]

        n_train = len(train_loader.dataset)
        self._update_state_dict(n_train=n_train)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if verbose:
            print(f"Training on {n_train} samples")
            print(
                f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}."
            )
            sys.stdout.flush()

    def on_epoch_start(self, epoch):
        self._update_state_dict(epoch=epoch)

    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)

    def on_before_loss(self, out, **kwargs):
        if (
            self.state_dict["epoch"] == 0
            and self.state_dict["idx"] == 0
            and self.state_dict["verbose"]
        ):
            print(f"Raw outputs of size {out.shape=}")

    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        # track training err and val losses to print at interval epochs
        msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
        values_to_log = dict(train_err=train_err, time=time, avg_loss=avg_loss)
        self._update_state_dict(msg=msg, values_to_log=values_to_log)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)
        self.loss.append(avg_loss)

    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            if isinstance(loss_value, float):
                self.state_dict["msg"] += f", {loss_name}={loss_value:.4f}"
            else:
                loss_value = {i: e.item() for (i, e) in enumerate(loss_value)}
                self.state_dict["msg"] += f", {loss_name}={loss_value}"
            self.state_dict["values_to_log"][loss_name] = loss_value
            self.val_loss.append(loss_value)
        
            

    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get("regularizer", False):
            avg_lasso = self.state_dict.get("avg_lasso_loss", 0.0)
            avg_lasso /= self.state_dict.get("n_epochs")
            self.state_dict["msg"] += f", avg_lasso={avg_lasso:.5f}"

        print(self.state_dict["msg"])
        sys.stdout.flush()

        if self.state_dict.get("wandb_log", False):
            for pg in self.state_dict["optimizer"].param_groups:
                lr = pg["lr"]
                self.state_dict["values_to_log"]["lr"] = lr
            wandb.log(
                self.state_dict["values_to_log"],
                step=self.state_dict["epoch"] + 1,
                commit=True,
            )
device = 'cpu'


# %%

x_train = torch.load('../../data_poisson/synthetic_data_dirichlet_1000_x.pt').type(torch.float32).clone()
y_train = torch.load('../../data_poisson/synthetic_data_dirichlet_1000_y.pt').clone()
y_train = torch.unsqueeze(y_train,1).to(device)
grid_boundaries=[[0, 1], [0, 1]]

x_test = torch.load('../../data_poisson/synthetic_data_dirichlet_test_100_x2.pt').type(torch.float32).clone()
y_test = torch.load('../../data_poisson/synthetic_data_dirichlet_test_100_y2.pt').clone()
y_test = torch.unsqueeze(y_test,1).to(device)

#data processing on training set
reduce_dims = list(range(x_train.ndim))
input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
input_encoder.fit(x_train)


reduce_dims = list(range(y_train.ndim))
output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
output_encoder.fit(y_train)

train_db = TensorDataset(
        x_train,
        y_train,
    )

train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

test_db = TensorDataset(
        x_test,
        y_test,
    )

test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
test_loaders = {100:test_loader}

pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
data_processor = DefaultDataProcessor(
        in_normalizer=None,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
data_processor = data_processor.to(device)

# %%
# We create a tensorized FNO model

model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  callbacks = [BasicLoggerCallback2()],
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[100].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))
    out,_ = data_processor.postprocess(out,data)

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()


print()
# %%
a = test_samples[6]
data = data_processor.preprocess(a, batched=False)
x = data['x']
y = data["y"]
out = model(x.unsqueeze(0))

fig, axs = plt.subplots(nrows = 1, ncols = 2)
fig.set_figwidth(15)

fig1 = axs[0].pcolormesh(y.squeeze().detach().numpy())
fig.colorbar(fig1)


fig1 = axs[1].pcolormesh(out.squeeze().detach().numpy())
fig.colorbar(fig1)
# %



# %%
a = torch.load("../../finput.pt").type(torch.float32).clone().to(device)

a = {"x": a, "y":a}

data = data_processor.preprocess(a, batched=False)
x = data['x']
y = data["y"]
out = model(x.unsqueeze(0))

fig, axs = plt.subplots(nrows = 1, ncols = 2)
fig.set_figwidth(15)

fig1 = axs[0].pcolormesh(y.squeeze().detach().numpy())
fig.colorbar(fig1)


fig1 = axs[1].pcolormesh(out.squeeze().detach().numpy())
fig.colorbar(fig1)
# %%
