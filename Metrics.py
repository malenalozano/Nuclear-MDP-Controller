# Import required dependencies
import numpy as np

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Absolute Error (MAE) """
    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Squared Error (MSE) """
    return np.mean((y_true - y_pred) ** 2)

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the R2 metric """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot)

def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Pearson's Correlation Coefficient """
    # np.corrcoef usa una convención coherente para varianzas y covarianza,
    # evitando inconsistencias entre np.cov (ddof=1) y np.std (ddof=0)
    # que harían que la correlación pudiera salirse del intervalo [-1, 1].
    return np.corrcoef(y_true, y_pred)[0, 1]
