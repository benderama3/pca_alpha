# PCA Alpha with outliers and a linear model

This script fits model with PCAAlpha and a Logistic Regression

```bash
python pca_alpha_linear.py \
    --add_outliers \
    --dataset digits \
    --fit_latent_space \
    --outlier_prob 0.01 \
    --outlier_factor 10 \
    --plot_name digits.png \
    --seed 123
```

* add_outliers (optional): adds outliers on the test set
* dataset (optional): name of the dataset
* fit_latent_space (optional): fit the linear model in the latent space
* outlier_prob (optional): probability for a feature to be an outlier
* outlier_factor (optional): factor by which an outlier feature is multiplied
* plot_name (optional): name of plot
* seed (optional): random seed