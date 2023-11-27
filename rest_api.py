import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.visualization import visualize_Gaussian_Porcess_with_product_kernal, visualize_Gaussian_Porcess_with_sum_kernal, visualize_Gaussian_Porcess_with_RBF_kernal, visualize_Gaussian_Porcess_with_gaussian_kernal, visualize_Gaussian_Porcess_with_RQ_kernal, visualize_Gaussian_Porcess_with_product_kernal_ap, visualize_Gaussian_Porcess_with_rbf_kernal_ap, visualize_Gaussian_Porcess_with_sum_kernal_ap, visualize_Gaussian_Porcess_with_gaussian_kernal_ap, visualize_Gaussian_Porcess_with_RQ_kernal_ap
from src.utils.evaluation import evaluate_Gaussian_Porcess_with_sum_kernal, evaluate_Gaussian_Porcess_with_product_kernal, evaluate_Gaussian_Porcess_with_RBF_kernal, evaluate_Gaussian_Porcess_with_RQ_kernal, evaluate_Gaussian_Porcess_with_gaussian_kernal, evaluate_Gaussian_Porcess_with_product_kernal_ap, evaluate_Gaussian_Porcess_with_rbf_kernal_ap, evaluate_Gaussian_Porcess_with_sum_kernal_ap, evaluate_Gaussian_Porcess_with_rq_kernal_ap, evaluate_Gaussian_Porcess_with_gaussian_kernal_ap
from src.models.kernals import GaussianKernel, SumKernel, RBFKernel, ProductKernel, RationalQuadraticKernel
from src.data.data_loader import load_mauna_loa_atmospheric_co2, load_international_airline_passengers
from src.models.GaussianProcess import GaussianProcess

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")  # Create a "templates" directory and put your HTML files in it


# Create instances of the kernel classes
gaussian_kernel = GaussianKernel(length_scale=1.0)
rbf_kernel = RBFKernel(length_scale=1.0)
rq_kernel = RationalQuadraticKernel(alpha=1.0, length_scale=1.0)
sum_kernel = SumKernel(kernel1=gaussian_kernel, kernel2=rbf_kernel)
prod_kernel = ProductKernel(gaussian_kernel, rbf_kernel)

# Load data
X_airpassengers, y_airpassengers, X_airpassengers_normalized = load_international_airline_passengers(
    'data/international-airline-passengers.csv')
X, y, X_normalized = load_mauna_loa_atmospheric_co2(
    'data/mauna_loa_atmospheric_co2.csv')


evaluation_results_mlc = {}
evaluation_results_ap = {}
plots_co2 = {}
plots_ap = {}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.get("/train", response_class=HTMLResponse)
def train_gaussian_process(request: Request):
    global gp_gaussian, gp_rbf, gp_rq, gp_sum, gp_prod, gp_sum_airpassengers, gp_prod_airpassengers, gp_rbf_airpassengers, gp_gaussian_airpassengers, gp_rq_airpassengers



    # Create instances of the GaussianProcess class with different kernels
    gp_gaussian = GaussianProcess(kernel=gaussian_kernel, noise=1e-5)
    gp_rbf = GaussianProcess(kernel=rbf_kernel, noise=1e-5)
    gp_rq = GaussianProcess(kernel=rq_kernel, noise=1e-5)
    gp_sum = GaussianProcess(kernel=sum_kernel, noise=1e-5)
    gp_prod = GaussianProcess(kernel=prod_kernel, noise=1e-5)

    gp_sum_airpassengers = GaussianProcess(kernel=sum_kernel, noise=1e-5)
    gp_prod_airpassengers = GaussianProcess(kernel=prod_kernel, noise=1e-5)
    gp_rbf_airpassengers = GaussianProcess(kernel=rbf_kernel, noise=1e-5)
    gp_gaussian_airpassengers = GaussianProcess(kernel=gaussian_kernel, noise=1e-5)
    gp_rq_airpassengers = GaussianProcess(kernel=rq_kernel, noise=1e-5)

    # Fit the Gaussian Processes
    gp_gaussian.fit(X_normalized, y)
    gp_rbf.fit(X_normalized, y)
    gp_rq.fit(X_normalized, y)
    gp_sum.fit(X_normalized, y)
    gp_prod.fit(X_normalized, y)

    gp_sum_airpassengers.fit(X_airpassengers_normalized, y_airpassengers)
    gp_prod_airpassengers.fit(X_airpassengers_normalized, y_airpassengers)
    gp_rbf_airpassengers.fit(X_airpassengers_normalized, y_airpassengers)
    gp_gaussian_airpassengers.fit(X_airpassengers_normalized, y_airpassengers)
    gp_rq_airpassengers.fit(X_airpassengers_normalized, y_airpassengers)

    return templates.TemplateResponse("training.html", {"request": request})

@app.get("/predict_mauna_loa_atmospheric_co2", response_class=HTMLResponse)
def predict_gaussian_process_mauna_loa_atmospheric_co2(request: Request):
    X_pred_normalized = np.linspace(X_normalized.min(), X_normalized.max(), 1000).reshape(-1, 1)

    X_pred = X_pred_normalized * X.std() + X.mean()

    y_pred_mean_prod, y_pred_cov_prod = gp_prod.predict(X_pred_normalized)
    y_pred_mean_gaussian, y_pred_cov_gaussian = gp_gaussian.predict(X_pred_normalized)
    y_pred_mean_rbf, y_pred_cov_rbf = gp_rbf.predict(X_pred_normalized)
    y_pred_mean_rq, y_pred_cov_rq = gp_rq.predict(X_pred_normalized)
    y_pred_mean_sum, y_pred_cov_sum = gp_sum.predict(X_pred_normalized)


    evaluation_results_mlc["gp_gaussian"] = evaluate_Gaussian_Porcess_with_gaussian_kernal(y,y_pred_mean_gaussian)
    evaluation_results_mlc["gp_prod"] = evaluate_Gaussian_Porcess_with_product_kernal(y, y_pred_mean_prod)
    evaluation_results_mlc["gp_rbf"] = evaluate_Gaussian_Porcess_with_RBF_kernal(y, y_pred_mean_rbf)
    evaluation_results_mlc["gp_rq"] = evaluate_Gaussian_Porcess_with_RQ_kernal(y, y_pred_mean_rq)
    evaluation_results_mlc["gp_sum"] = evaluate_Gaussian_Porcess_with_sum_kernal(y, y_pred_mean_sum)

    #Create visualizations as jbg files in output/co2_concentration
    plots_co2["gp_gaussian"] = visualize_Gaussian_Porcess_with_gaussian_kernal(X,y,X_pred, y_pred_mean_gaussian, y_pred_cov_gaussian)
    plots_co2["gp_prod"] = visualize_Gaussian_Porcess_with_product_kernal(X,y,X_pred, y_pred_mean_prod, y_pred_cov_prod)
    plots_co2["gp_rbf"] = visualize_Gaussian_Porcess_with_RBF_kernal(X,y,X_pred, y_pred_mean_rbf, y_pred_cov_rbf)
    plots_co2["gp_rq"] = visualize_Gaussian_Porcess_with_RQ_kernal(X,y,X_pred, y_pred_mean_rq, y_pred_cov_rq)
    plots_co2["gp_sum"] = visualize_Gaussian_Porcess_with_sum_kernal(X,y,X_pred, y_pred_mean_sum, y_pred_cov_sum)

    return templates.TemplateResponse("prediction_mauna_loa_atmospheric_co2.html", {
        "request": request,
        "evaluation_results" : evaluation_results_mlc,
        "plots": plots_co2
    })
@app.get("/predict_international_airline_passengers", response_class=HTMLResponse)
def predict_gaussian_process_international_airline_passengers(request: Request):
    X_pred_airpassengers_normalized = np.linspace(X_airpassengers_normalized.min(), X_airpassengers_normalized.max(),
                                                  1000).reshape(-1, 1)
    X_pred_airpassengers = (X_pred_airpassengers_normalized * X_airpassengers.std() + X_airpassengers.mean()).astype(
        int)

    y_pred_mean_airpassengers_sum, y_pred_cov_airpassengers_sum = gp_sum_airpassengers.predict(
        X_pred_airpassengers_normalized)
    
    y_pred_mean_airpassengers_rbf, y_pred_cov_airpassengers_rbf = gp_rbf_airpassengers.predict(
        X_pred_airpassengers_normalized)
    y_pred_mean_airpassengers_gaussian, y_pred_cov_airpassengers_gaussian = gp_gaussian_airpassengers.predict(
        X_pred_airpassengers_normalized)
    y_pred_mean_airpassengers_rq, y_pred_cov_airpassengers_rq = gp_rq_airpassengers.predict(
        X_pred_airpassengers_normalized)
    y_pred_mean_airpassengers_prod, y_pred_cov_airpassengers_prod = gp_prod_airpassengers.predict(
        X_pred_airpassengers_normalized)    
        
        
    evaluation_results_ap["gp_rq_airpassengers"] = evaluate_Gaussian_Porcess_with_product_kernal_ap(y, y_pred_mean_airpassengers_prod)
    evaluation_results_ap["gp_sum_airpassengers"] = evaluate_Gaussian_Porcess_with_gaussian_kernal_ap(y, y_pred_mean_airpassengers_gaussian)
    evaluation_results_ap["gp_prod_airpassengers"] = evaluate_Gaussian_Porcess_with_rbf_kernal_ap(y, y_pred_mean_airpassengers_rbf)
    evaluation_results_ap["gp_rbf_airpassengers"] = evaluate_Gaussian_Porcess_with_rq_kernal_ap(y, y_pred_mean_airpassengers_rq)
    evaluation_results_ap["gp_gaussian_airpassengers"] =evaluate_Gaussian_Porcess_with_sum_kernal_ap(y, y_pred_mean_airpassengers_sum)

    #visualize
    plots_ap["gp_rq_airpassengers"] = visualize_Gaussian_Porcess_with_RQ_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers_rq,y_pred_cov_airpassengers_rq)
    plots_ap["gp_rbf_airpassengers"] = visualize_Gaussian_Porcess_with_rbf_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers_rbf,y_pred_cov_airpassengers_rbf)
    plots_ap["gp_sum_airpassengers"] = visualize_Gaussian_Porcess_with_sum_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers_sum,y_pred_cov_airpassengers_sum)
    plots_ap["gp_gaussian_airpassengers"] = visualize_Gaussian_Porcess_with_gaussian_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers_gaussian,y_pred_cov_airpassengers_gaussian)
    plots_ap["gp_prod_airpassengers"] = visualize_Gaussian_Porcess_with_product_kernal_ap(X_pred_airpassengers,y_pred_mean_airpassengers_prod,y_pred_cov_airpassengers_prod)

    return templates.TemplateResponse("prediction_international_airline_passengers.html", {"request": request, "evaluation_results": evaluation_results_ap, "plots": plots_ap})
