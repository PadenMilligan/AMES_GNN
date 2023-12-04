""" Defines fit_model for training a model """

from typing import List

import torch
from torch_geometric.loader import DataLoader



def fit_model(
    n_epochs: int,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_train: int,
    n_validation: int,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    n_epoch_0: int = 0,
    calculate_forces: bool = False,
    weight: float = 1.0,
    writer: torch.utils.tensorboard.SummaryWriter = None,
    callbacks: List[object] = None,
    check_point_path: str = "chkpt.tar",
    check_point_freq: int = 10,
) -> None:

    """

    This function centralises the optimization of the model parameters

    Args:

    :param: n_epochs: the number of epochs (steps) to perform in the minimization
    :type: int
    :param: model: the model being optimized
    :type: torch.nn.Module
    :param: loss_func: the loss function that is minimized during the process
    :type: torch.nn.Module
    :param: optimizer: the optimization strategy used to minimize the loss
    :type: torch.optim.Optimizer
    :param: n_train: the number of samples in the training dataset
    :type: int
    :param: n_validation: the number of samples in the validation dataset
    :type: int
    :param: train_dl: the data-loader for training
    :type: torch_geometric.loader.DataLoader
    :param: valid_dl: the data-loader for validation
    :type: torch_geometric.loader.DataLoader
    :param: n_epoch_0: the initial epoch index (different from 0 if this
           is a continuation run)
    :type: int
    :param: calculate_forces: True if forces need to be calculated and
           fitted against (default false)
    :type: bool
    :param: weight: if forces are used, the weight to be given to
           forces vs energy (1 means equal weight)
    :type: float
    :param: writer: TensorBoard writer to document the run
    :type: torch.utils.tensorboard.SummaryWriter
    :param: callbacks: list of callbacks to be used (e.g. early-stopping, user-stop, etc)
    :type: List[callback]
    :param: check_point_path: filename path to save model information during
           checkpointing or callback-stopping.
    :type: str
    :param: check_point_freq: model is checkpointed every check_point_freq
           epochs (steps)
    :type: int

    """

    factor = float(n_train) / float(n_validation)

    model.train()

    for epoch in range(n_epochs):

        n_epoch = n_epoch_0 + epoch

        train_running_loss = 0.0

        for sample in train_dl:

            optimizer.zero_grad()

            energy = model(
                sample.x, sample.edge_index, sample.edge_attr, sample.batch
            )

            train_loss = loss_func(torch.squeeze(energy, dim=1), sample.y)

            train_loss.backward()

            optimizer.step()

            train_running_loss += train_loss.item()

        val_running_loss = 0.0

        # with torch.set_grad_enabled( False ):

        for sample in valid_dl:

            torch.set_grad_enabled(False)

            energy = model(
                sample.x, sample.edge_index, sample.edge_attr, sample.batch
            )

            val_loss = loss_func(torch.squeeze(energy, dim=1), sample.y)

            torch.set_grad_enabled(True)

            val_running_loss += val_loss.item()

        val_running_loss *= (
            factor  # to put it on the same scale as the training running loss
        )

        print(
            repr(n_epoch) + ",  " + repr(train_running_loss) + ",  " + repr(val_running_loss)
        )

        if writer is not None:

            writer.add_scalar("training loss", train_running_loss, n_epoch)
            writer.add_scalar("validation loss", val_running_loss, n_epoch)

            for name, weight in model.named_parameters():
                writer.add_histogram( name, weight, epoch )
                writer.add_histogram( f'{name}.grad', weight.grad, epoch )

        # if we should store the current state of mode/optimizer, do so here

        if (
            n_epoch + 1
        ) % check_point_freq == 0:  # n_epoch + 1 to ensure saving at the last iteration too

            torch.save(
                {
                    "epoch": n_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_running_loss,
                    "val_loss": val_running_loss,
                },
                check_point_path,
            )

        # if there are any callbacks, act them if needed

        for callback in callbacks:

            callback(train_running_loss)

            # check for early stopping; if true, we return to main function

            if (
                callback.early_stop
            ):  # if we are to stop, make sure we save model/optimizer

                torch.save(
                    {
                        "epoch": n_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_running_loss,
                        "val_loss": val_running_loss,
                    },
                    check_point_path,
                )

                return
