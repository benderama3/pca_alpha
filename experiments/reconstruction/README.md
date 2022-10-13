# PCA Alpha with outliers and a linear model

This script fits model with PCAAlpha and a Logistic Regression

```bash
python pca_alpha_reconstruction.py \
    --add_outliers \
    --dataset digits \
    --device cuda \
    --plot_name digits.png \
    --seed 123
```

* add_outliers (optional): adds outliers on the test set
* dataset (optional): name of the dataset
* device (optional): device
* plot_name (optional): name of plot
* seed (optional): random seed