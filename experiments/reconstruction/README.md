# Reconstruction loss on PCA Alpha

This script computes MAE on reconstructed images

```bash
python pca_alpha_reconstruction.py \
    --add_outliers \
    --dataset mnist \
    --device cuda \
    --max_samples 500 \
    --plot_name digits.png \
    --seed 123
```

* add_outliers (optional): adds outliers on the test set
* dataset (optional): name of the dataset
* device (optional): device
* max_samples (optional): max samples to use
* plot_name (optional): name of plot
* seed (optional): random seed
