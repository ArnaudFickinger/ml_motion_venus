# Optimal Prediction with “Mayan Data”

## Kernel Ridge Regression
To reproduce the experiments on Kernel Ridge Regression with Venus motion data, open the Jupyter notebook "kernel_ridge_motion_venus.ipynb" and run the cells one by one. By doing so you will:
- get data from HORIZON system of NASA.
- sparsify the data and add noise to it to mimic the mayan observation.
- perform RBF kernel ridge regression and plot out the fitting results.
- perform RBF kernel ridge regression on fourier features and plot out the fitting results.

## Modelling the learning process of the Mayan with Meta-Learning
To reproduce the two experiments of this part, open the Jupyter notebook "meta_learning_motion_venus.ipynb" and run all the blocks. You can choose between the first and the second experiment by changing the last line of the last block:
```
plot_true_and_predicted(seed=0, d=d, n_train_inner=300,n_train_meta=2500, tasks_prob=(1/3,1/3,1/3,0), num_iterations=101, num_inner_tasks=10)

```
```
plot_true_and_predicted(seed=0, d=d, n_train_inner=300,n_train_meta=2500, tasks_prob=(1/2,1/2,0,0), num_iterations=101, num_inner_tasks=10)

```
## Ridge Regression and LASSO

To reproduce the experiments on Ridge Regression and LASSO, open the notebook "ridge_lasso_motion_venus.ipynb" and follow the instructions. The parameters used for the Ridge Regression experiment:

```
X_mat = featurize_fourier(x_train, d, L)
weight = solve_ridge(X_mat,y_train,lambda_ridge=lamb)
```
are:

* d = 611
* L = 350668 hours
* lamb = 10

The paramters used for the LASSO experiment:

```
X_mat = featurize_fourier(x_train, d, L)
weight = solve_lasso(X_mat,y_train,lambda_lasso=lamb)
```
are:
* d = 611
* L = 350668 hours
* lamb = 10**-3
