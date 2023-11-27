
from sklearn.metrics import mean_squared_error
from math import sqrt


def evaluate_Gaussian_Porcess_with_sum_kernal(y,y_pred_mean_sum ):
    mse = mean_squared_error(y[:788], y_pred_mean_sum[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the addition kernals: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_product_kernal(y,y_pred_mean_prod ):
    mse = mean_squared_error(y[:788], y_pred_mean_prod[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the Multiple kernals: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_gaussian_kernal(y,y_pred_mean_gaussian ):
    mse = mean_squared_error(y[:788], y_pred_mean_gaussian[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the Gaussian kernals: {rmse:.2f}')

    return rmse


def evaluate_Gaussian_Porcess_with_RQ_kernal(y,y_pred_mean_rq ):
    mse = mean_squared_error(y[:788], y_pred_mean_rq[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the RQ kernal: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_RBF_kernal(y,y_pred_mean_rbf ):
    mse = mean_squared_error(y[:788], y_pred_mean_rbf[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the RBF kernals: {rmse:.2f}')
    return rmse


def evaluate_Gaussian_Porcess_with_sum_kernal_ap(y,y_pred_mean_airpassengers_sum ):
    mse = mean_squared_error(y[:788], y_pred_mean_airpassengers_sum[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the addition kernal with air_passenger: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_product_kernal_ap(y,y_pred_mean_airpassengers_prod ):
    mse = mean_squared_error(y[:788], y_pred_mean_airpassengers_prod[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the Multiple  kernal with air_passenger: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_gaussian_kernal_ap(y,y_pred_mean_airpassengers_gaussian ):
    mse = mean_squared_error(y[:788], y_pred_mean_airpassengers_gaussian[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the Guassian kernals with air_passenger: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_rbf_kernal_ap(y,y_pred_mean_airpassengers_rbf ):
    mse = mean_squared_error(y[:788], y_pred_mean_airpassengers_rbf[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the RBF kernals with air_passenger: {rmse:.2f}')
    return rmse

def evaluate_Gaussian_Porcess_with_rq_kernal_ap(y,y_pred_mean_airpassengers_rq ):
    mse = mean_squared_error(y[:788], y_pred_mean_airpassengers_rq[:788])
    rmse = sqrt(mse)
    print(f'RMSE for the RQ kernals with air_passenger: {rmse:.2f}')
    return rmse
