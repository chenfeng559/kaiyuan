<<<<<<< HEAD
import numpy as np

=======
import mindspore.numpy as np
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

<<<<<<< HEAD

=======
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

<<<<<<< HEAD

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / np.mean(np.abs(true))


def WAPE(pred, true):
    return np.mean(np.abs(pred - true)) / np.mean(np.abs(true))



=======
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / np.mean(np.abs(true))

def WAPE(pred, true):
    return np.mean(np.abs(pred - true)) / np.mean(np.abs(true))

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
