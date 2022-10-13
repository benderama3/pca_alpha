import torch 
import numpy as np
from pca_alpha import *
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

def process_dataset(dataset, extractor, is_rgb=False):
    dataset_train = dataset["train"].shuffle(seed=SEED)[:max_samples]
    try:
        dataset_test = dataset["test"].shuffle(seed=SEED)[:max_samples]
    except:
        dataset_test = dataset["validation"].shuffle(seed=SEED)[:max_samples]

    X_train, y_train = dataset_train[image_name], np.array(dataset_train[label_name])
    base_X_test, y_test = dataset_test[image_name], np.array(dataset_test[label_name])

    if not is_rgb:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="np")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="np")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
    else:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="np")["pixel_values"][:max_samples].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="np")["pixel_values"][:max_samples].reshape(max_samples, -1)
    return X_train, y_train, base_X_test, y_test


parser = argparse.ArgumentParser()

parser.add_argument('--add_outliers', action='store_true')
parser.add_argument('--fit_latent_space', action='store_true')
parser.add_argument('--dataset', type=str, required=False, default="digits")
parser.add_argument('--max_samples', type=int, required=False, default=500)
parser.add_argument('--outlier_prob', type=float, required=False, default=0.01)
parser.add_argument('--outlier_factor', type=float, required=False, default=10)
parser.add_argument('--plot_name', type=str, required=False, default="plot.png")
parser.add_argument('--seed', type=int, required=False, default=123)
args = parser.parse_args()



if __name__ == "__main__":

    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=16)
    plt.rcParams["figure.figsize"] = (8, 5)

    dataset = args.dataset
    assert dataset in ['digits', 'diabetes', 'wine', 'fashion_mnist', 'mnist'], "--dataset must be in ['digits', 'diabetes', 'wine', 'fashion_mnist', 'mnist']"

    SEED = args.seed
    max_samples = args.max_samples
    plot_name = args.plot_name
    path = "plot/" + dataset + "/"

    alphas = [0.5, 0.75, 1., 1.25, 1.5]

    if not os.path.exists(path):
        os.makedirs(path)

    if dataset == "digits":
        n_components = [1, 2, 4, 8, 16, 24, 32]
        is_regression = False
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        X_train, base_X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)

    elif dataset == "mnist":
        n_components = [1, 2, 4, 8, 16, 24, 32]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset
        
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "mnist", "farleyknight-org-username/vit-base-mnist"
        extractor = ViTFeatureExtractor(do_resize=False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        #extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=False)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    elif dataset == "fashion_mnist":
        n_components = [1, 2, 4, 8, 16, 24, 32]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset
        
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "fashion_mnist", "abhishek/autotrain_fashion_mnist_vit_base"
        extractor = ViTFeatureExtractor(do_resize=False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        #extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=False)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    elif dataset == "wine":
        n_components = [1, 2, 3, 4, 5, 6, 7, 8]
        is_regression = False
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        X_train, base_X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)

    elif dataset == "diabetes":
        n_components = [1, 2, 3, 4, 5, 6, 7, 8]
        is_regression = True
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        X_train, base_X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)
    else:
        raise()

    if args.add_outliers:
        n, d = base_X_test.shape
        outliers = np.random.binomial(1, args.outlier_prob, n*d).reshape(n, d)
        base_X_test += (args.outlier_factor - 1) * base_X_test * outliers

    for alpha in alphas:
        scores = []

        if not is_regression:
            base_model = LogisticRegression(max_iter=500, random_state=SEED).fit(X_train, y_train)
        else:
            base_model = LinearRegression().fit(X_train, y_train)

        for n_comp in n_components:
            
            if args.fit_latent_space:
                pca = PCAAlpha(n_components=n_comp, alpha=alpha, random_state=SEED)
                pca.fit(X_train)

                if not is_regression:
                    base_model = LogisticRegression(max_iter=500, random_state=SEED).fit(pca.transform(X_train), y_train)
                else:
                    base_model = LinearRegression().fit(pca.transform(X_train), y_train)

                X_test = pca.transform(base_X_test)
                accuracy = base_model.score(X_test, y_test)
            else:
                pca = PCAAlpha(n_components=n_comp, alpha=alpha, random_state=SEED)
                pca.fit(X_train)
                X_test = pca.approximate(base_X_test)
                accuracy = base_model.score(X_test, y_test)

            print("n_comp:", n_comp, "  alpha:", alpha, "  accuracy:", accuracy)
            scores.append(accuracy)

        plt.plot(n_components, scores, label="alpha: " + str(alpha))
    plt.legend(loc="best")
    plt.xlabel("Number of components")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path + plot_name, dpi=250)