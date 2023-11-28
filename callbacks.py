import os
from os.path import exists
import re
from typing import Dict, List

import torch
from torch.optim import Optimizer
import yaml


class LRScheduler:

    """
    Learning rate scheduler. If the validation loss does not decrease
    for the given number of `patience` epochs, then the learning rate
    will decrease by by given `factor`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        patience: int = 5,
        min_lr: float = 1.0e-6,
        factor: float = 0.5,
    ) -> None:

        r"""

        Updates the learning rate as:
        new_lr = old_lr * factor

        Args:

        :param: optimizer (Optimizer): the optimizer we are using patience
        :type: torch.optim.Optimizer
        :param: patience (int) how many epochs to wait before updating the lr
        :type: int
        :param: min_lr (float): least lr value to reduce to while updating
        :type: float
        :param: factor (float): factor by which the lr should be updated
        :type: float

        """

        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        self.lr_scheduler.step(val_loss)


class EarlyStopping:

    """
    Early stopping to stop the training when the loss does not
    improve after certain epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1.0e-4) -> None:

        """

        Args:

        :param: patience (int): how many epochs to wait before stopping
               when loss is not improving
        :type: int
        :param: min_delta (float): minimum difference between new loss
               and old loss for new loss to be considered as an
               improvement
        :type: float

        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:

        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} \
                  of {self.patience}"
            )
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


class UserStopping:

    """
    This callback allows user to specify a yaml file from which
    to read a stop flag, by default set to False; if the user sets the
    flag to True in the file, the program will receive a signal to
    stop; this gives the user the possibility to stop execution
    if convergence is slow or for whatever other practical reason.

    """

    def __init__(self) -> None:

        self.flag_file = "STOPFLAG.yml"
        self.early_stop = False

        # if the stopfile exists, remove it; it was probably left
        # over from a previous simulation

        if exists(self.flag_file):
            os.system("rm -f " + self.flag_file)

        # flag_file = open(self.flag_file, "wt")
        with open( self.flag_file, 'wt' ) as flag_file:
            flag_file.write("STOPFLAG: False")

    def __call__(self, val_loss: float) -> None:

        # stop_stream = open(self.flag_file, "rt")
        with open(self.flag_file, 'rt') as stop_stream:
            stream = yaml.load(stop_stream, Loader=yaml.Loader)
            self.early_stop = stream["STOPFLAG"]

        if self.early_stop:
            print("INFO: User instructed stopping")
            os.system("rm -f " + self.flag_file)
            # we remove the stop-flag file so that it is not
            # there when we next run the program


def set_up_callbacks(callback_list: Dict, optimizer: Optimizer) -> List:

    """
    Callback factory function

    :param: callback_list (dictonary of callbacks)
    :type: dict
    :param: optimizer on which callbacks are to act
    :type: torch.optim.Optimizer
    :rtype: List

    """

    callbacks = []

    names = callback_list.keys()

    sched = re.compile("sched", re.IGNORECASE)
    early = re.compile("early", re.IGNORECASE)
    ucontrol = re.compile("user", re.IGNORECASE)

    for name in names:

        callback = callback_list[name]

        if re.search(sched, name):  # set-up the LR scheduler

            patience = callback.get("patience", 10)
            min_lr = callback.get("min_lr", 1.0e-6)
            factor = callback.get("factor", 0.5)

            callbacks.append(
                LRScheduler(optimizer, patience=patience, min_lr=min_lr, factor=factor)
            )

        elif re.search(early, name):  # set-up early-stopping callback

            patience = callback.get("patience", 10)
            min_delta = callback.get("min_delta", 1.0e-6)

            callbacks.append(EarlyStopping(patience=patience, min_delta=min_delta))

        elif re.search(ucontrol, name):  # set-up user-stopping
            # callback; the user can set a flag in the STOPFLAG.yml

            callbacks.append(UserStopping())

    return callbacks
