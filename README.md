Under review

# Alpha-PCA

1 to 80 Components         |  16 Components            |  64 Components
:-------------------------:|:-------------------------:|:-------------------------:
<img src="gif/alpha_0.75.gif" width="108" height="192"/>  |  <img src="gif/alpha_16_0.75.gif" width="108" height="192"/>  |  <img src="gif/alpha_64_0.75.gif" width="108" height="192"/>

Alpha-PCA is more robust to outliers than standard PCA. \
Standard PCA is a special case of alpha PCA (when alpha=1).

* [Usage](#usage)
* [Experiments](#experiments)

## Usage

The model is inherited from a sklearn module and works the same way as the [standard PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). \
It also supports [PyTorch](https://pytorch.org/) tensors (on cpu and GPU).

```python
from pca_alpha import *
import torch 

X = torch.randn(16, 10) # also works with numpy
pca = PCAAlpha(n_components=5, alpha=0.7, random_state=123) # alpha=1 -> standard PCA
pca.fit(X)

# to project X in the latent space
X_transformed = pca.transform(X) # (16, 10) -> (16, 5)

# fit inverse
X_ = pca.inverse_transform(X_transformed) # (16, 5) -> (16, 10)

# directly approximate X_ == inverse_transform(transform(X))
X_ = pca.approximate(X) # (16, 10) -> (16, 10)
```

When the number of features exceed the number of samples, the model uses a transposed decomposition for efficiency.

## Experiments

Experiments are located in the `experiments/` folder.