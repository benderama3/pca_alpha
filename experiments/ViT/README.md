# PCA Alpha with a ViT model

This script fits model with PCAAlpha and a ViT model

```bash
python pca_alpha_vit.py \
    --add_noise \
    --dataset mnist \
    --device cuda \
    --max_samples 250 \
    --noise_std 0.25 \
    --plot_name mnist_with_noise.png \
    --seed 123
```

* add_noise (optional): add noise
* dataset (optional): name of the dataset
* device: device for evaluation
* max_samples (optional): number of samples to use
* noise_std (optional): noise standard deviation
* plot_name (optional): name of plot
* seed (optional): random seed

