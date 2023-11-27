import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


output_directory_co2 = 'output/co2_concentration'
output_directory_air_passenger = 'output/air_passenger'

def visualize_Gaussian_Porcess_with_sum_kernal(X, y, X_pred,y_pred_mean_sum, y_pred_cov_sum ):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Observations')
    plt.plot(X_pred, y_pred_mean_sum, color='purple', label='Prediction (Product Kernel)')
    plt.fill_between(X_pred.flatten(), y_pred_mean_sum - 1.96 * np.sqrt(np.diag(y_pred_cov_sum)),
                     y_pred_mean_sum + 1.96 * np.sqrt(np.diag(y_pred_cov_sum)), alpha=0.2, color='purple')
    plt.xlabel('Decimal Date')
    plt.ylabel('CO2 Concentration (ppm)')
    plt.title('Gaussian Process Regression with Product Kernels')
    plt.legend()


    output_file_path = f"{output_directory_co2}/gaussian_process_with_sum_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path


def visualize_Gaussian_Porcess_with_product_kernal(X, y, X_pred,y_pred_mean_prod, y_pred_cov_prod ):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Observations')
    plt.plot(X_pred, y_pred_mean_prod, color='purple', label='Prediction (Product Kernel)')
    plt.fill_between(X_pred.flatten(), y_pred_mean_prod - 1.96 * np.sqrt(np.diag(y_pred_cov_prod)),
                     y_pred_mean_prod + 1.96 * np.sqrt(np.diag(y_pred_cov_prod)), alpha=0.2, color='purple')
    plt.xlabel('Decimal Date')
    plt.ylabel('CO2 Concentration (ppm)')
    plt.title('Gaussian Process Regression with Product Kernels')
    plt.legend()


    output_file_path = f"{output_directory_co2}/gaussian_process_with_product_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path


def visualize_Gaussian_Porcess_with_gaussian_kernal(X, y, X_pred,y_pred_mean_gaussian, y_pred_cov_gaussian ):
    plt.figure(figsize=(15, 10))
    plt.scatter(X, y, color='red', label='Observations')
    plt.plot(X_pred, y_pred_mean_gaussian, color='blue', label='Prediction (Gaussian)')
    plt.fill_between(X_pred.flatten(), y_pred_mean_gaussian - 1.96 * np.sqrt(np.diag(y_pred_cov_gaussian)),
                     y_pred_mean_gaussian + 1.96 * np.sqrt(np.diag(y_pred_cov_gaussian)), alpha=0.2, color='blue')
    plt.xlabel('Decimal Date')
    plt.ylabel('CO2 Concentration (ppm)')
    plt.title('Gaussian Process Regression with Gaussian Kernel')
    plt.legend()


    output_file_path = f"{output_directory_co2}/gaussian_process_with_sum_gaussian_plot.png"
    plt.savefig(output_file_path)
    return output_file_path


def visualize_Gaussian_Porcess_with_RQ_kernal(X, y, X_pred,y_pred_mean_rq, y_pred_cov_rq ):
    plt.figure(figsize=(15, 10))
    plt.scatter(X, y, color='red', label='Observations')
    plt.plot(X_pred, y_pred_mean_rq, color='purple', label='Prediction (RQ)')
    plt.fill_between(X_pred.flatten(), y_pred_mean_rq - 1.96 * np.sqrt(np.diag(y_pred_cov_rq)),
                     y_pred_mean_rq + 1.96 * np.sqrt(np.diag(y_pred_cov_rq)), alpha=0.2, color='purple')
    plt.xlabel('Decimal Date')
    plt.ylabel('CO2 Concentration (ppm)')
    plt.title('Gaussian Process Regression with Rational Quadratic')
    plt.legend()


    output_file_path = f"{output_directory_co2}/gaussian_process_with_Rational_Quadratic_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path


def visualize_Gaussian_Porcess_with_RBF_kernal(X, y, X_pred,y_pred_mean_rbf, y_pred_cov_rbf ):
    plt.figure(figsize=(15, 10))
    plt.scatter(X, y, color='red', label='Observations')
    plt.plot(X_pred, y_pred_mean_rbf, color='green', label='Prediction (RBF)')
    plt.fill_between(X_pred.flatten(), y_pred_mean_rbf - 1.96 * np.sqrt(np.diag(y_pred_cov_rbf)),
                     y_pred_mean_rbf + 1.96 * np.sqrt(np.diag(y_pred_cov_rbf)), alpha=0.2, color='green')
    plt.xlabel('Decimal Date')
    plt.ylabel('CO2 Concentration (ppm)')
    plt.title('Gaussian Process Regression with RBF Kernel')
    plt.legend()


    output_file_path = f"{output_directory_co2}/gaussian_process_with_RBF_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path


def visualize_Gaussian_Porcess_with_sum_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers, y_pred_cov_airpassengers_sum ):
    df_airpassengers = pd.read_csv('data/international-airline-passengers.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df_airpassengers['Month'], df_airpassengers['Passengers'], color='red', label='Observations')

    # Convert X_pred_airpassengers to string format and plot
    X_pred_airpassengers_str = X_pred_airpassengers.astype(str)
    plt.plot(np.arange(len(X_pred_airpassengers_str)), y_pred_mean_airpassengers, color='purple',
             label='Prediction (Sum Kernel)')

    # Fill between the upper and lower confidence bounds
    plt.fill_between(np.arange(len(X_pred_airpassengers_str)),
                     y_pred_mean_airpassengers - 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_sum)),
                     y_pred_mean_airpassengers + 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_sum)), alpha=0.2,
                     color='purple')

    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Gaussian Process Regression with Sum Kernels (airpassengers)')

    # Set custom tick labels
    plt.xticks(np.arange(len(X_pred_airpassengers_str)), X_pred_airpassengers_str, ha="right")

    plt.legend()


    output_file_path = f"{output_directory_air_passenger}/gaussian_process_with_sum_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path


def visualize_Gaussian_Porcess_with_product_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers, y_pred_cov_airpassengers_prod ):
    df_airpassengers = pd.read_csv('data/international-airline-passengers.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df_airpassengers['Month'], df_airpassengers['Passengers'], color='red', label='Observations')

    # Convert X_pred_airpassengers to string format and plot
    X_pred_airpassengers_str = X_pred_airpassengers.astype(str)
    plt.plot(np.arange(len(X_pred_airpassengers_str)), y_pred_mean_airpassengers, color='purple',
             label='Prediction (Product Kernel)')

    # Fill between the upper and lower confidence bounds
    plt.fill_between(np.arange(len(X_pred_airpassengers_str)),
                     y_pred_mean_airpassengers - 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_prod)),
                     y_pred_mean_airpassengers + 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_prod)), alpha=0.2,
                     color='purple')

    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Gaussian Process Regression with Product Kernels (airpassengers)')

    # Set custom tick labels
    plt.xticks(np.arange(len(X_pred_airpassengers_str)), X_pred_airpassengers_str, ha="right")

    plt.legend()



    output_file_path = f"{output_directory_air_passenger}/gaussian_process_with_product_kernal_plot.jpg"

    plt.savefig(output_file_path, format='jpg', bbox_inches='tight', pad_inches=0.1)

    return output_file_path

def visualize_Gaussian_Porcess_with_gaussian_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers, y_pred_cov_airpassengers_gaussian ):
    df_airpassengers = pd.read_csv('data/international-airline-passengers.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df_airpassengers['Month'], df_airpassengers['Passengers'], color='red', label='Observations')

    # Convert X_pred_airpassengers to string format and plot
    X_pred_airpassengers_str = X_pred_airpassengers.astype(str)
    plt.plot(np.arange(len(X_pred_airpassengers_str)), y_pred_mean_airpassengers, color='purple',
             label='Prediction (gaussian Kernel)')

    # Fill between the upper and lower confidence bounds
    plt.fill_between(np.arange(len(X_pred_airpassengers_str)),
                     y_pred_mean_airpassengers - 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_gaussian)),
                     y_pred_mean_airpassengers + 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_gaussian)), alpha=0.2,
                     color='purple')

    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Gaussian Process Regression with gaussian Kernels (airpassengers)')

    # Set custom tick labels
    plt.xticks(np.arange(len(X_pred_airpassengers_str)), X_pred_airpassengers_str, ha="right")

    plt.legend()

    output_file_path = f"{output_directory_air_passenger}/gaussian_process_with_sum_gaussian_plot.png"
    plt.savefig(output_file_path)
    return output_file_path

def visualize_Gaussian_Porcess_with_rbf_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers, y_pred_cov_airpassengers_rbf ):
    df_airpassengers = pd.read_csv('data/international-airline-passengers.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df_airpassengers['Month'], df_airpassengers['Passengers'], color='red', label='Observations')

    # Convert X_pred_airpassengers to string format and plot
    X_pred_airpassengers_str = X_pred_airpassengers.astype(str)
    plt.plot(np.arange(len(X_pred_airpassengers_str)), y_pred_mean_airpassengers, color='purple',
             label='Prediction (rbf Kernel)')

    # Fill between the upper and lower confidence bounds
    plt.fill_between(np.arange(len(X_pred_airpassengers_str)),
                     y_pred_mean_airpassengers - 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_rbf)),
                     y_pred_mean_airpassengers + 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_rbf)), alpha=0.2,
                     color='purple')

    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Gaussian Process Regression with rbf Kernels (airpassengers)')

    # Set custom tick labels
    plt.xticks(np.arange(len(X_pred_airpassengers_str)), X_pred_airpassengers_str, ha="right")

    plt.legend()


    output_file_path = f"{output_directory_air_passenger}/gaussian_process_with_RBF_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path

def visualize_Gaussian_Porcess_with_RQ_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers, y_pred_cov_airpassengers_rq ):
    df_airpassengers = pd.read_csv('data/international-airline-passengers.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df_airpassengers['Month'], df_airpassengers['Passengers'], color='red', label='Observations')

    # Convert X_pred_airpassengers to string format and plot
    X_pred_airpassengers_str = X_pred_airpassengers.astype(str)
    plt.plot(np.arange(len(X_pred_airpassengers_str)), y_pred_mean_airpassengers, color='purple',
             label='Prediction (rq Kernel)')

    # Fill between the upper and lower confidence bounds
    plt.fill_between(np.arange(len(X_pred_airpassengers_str)),
                     y_pred_mean_airpassengers - 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_rq)),
                     y_pred_mean_airpassengers + 1.96 * np.sqrt(np.diag(y_pred_cov_airpassengers_rq)), alpha=0.2,
                     color='purple')

    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Gaussian Process Regression with rq Kernels (airpassengers)')

    # Set custom tick labels
    plt.xticks(np.arange(len(X_pred_airpassengers_str)), X_pred_airpassengers_str, ha="right")

    plt.legend()


    output_file_path = f"{output_directory_air_passenger}/gaussian_process_with_Rational_Quadratic_kernal_plot.png"
    plt.savefig(output_file_path)
    return output_file_path