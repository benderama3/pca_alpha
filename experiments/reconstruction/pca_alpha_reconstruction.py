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
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="pt")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="pt")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
    else:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="pt")["pixel_values"][:max_samples].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="pt")["pixel_values"][:max_samples].reshape(max_samples, -1)
    return X_train, y_train, base_X_test, y_test

parser = argparse.ArgumentParser()

parser.add_argument('--add_outliers', action='store_true')
parser.add_argument('--dataset', type=str, required=False, default="digits")
parser.add_argument('--device', type=str, required=False, default="cpu")
parser.add_argument('--max_samples', type=int, required=False, default=500)
parser.add_argument('--plot_name', type=str, required=False, default="plot.png")
parser.add_argument('--seed', type=int, required=False, default=123)
args = parser.parse_args()



if __name__ == "__main__":
    
    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=16)
    plt.rcParams["figure.figsize"] = (8, 5)

    dataset = args.dataset
    assert dataset in ['fashion_mnist', 'mnist'], "--dataset must be in ['fashion_mnist', 'mnist']"

    SEED = args.seed
    max_samples = args.max_samples
    plot_name = args.plot_name
    path = "plot/" + dataset + "/"

    alphas = [0.5, 0.75, 1., 1.25, 1.5]

    if not os.path.exists(path):
        os.makedirs(path)

    if dataset == "mnist":
        n_components = [1, 2, 4, 8, 16, 24, 32, 64]
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
        n_components = [1, 2, 4, 8, 16, 24, 32, 64]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset
        
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "fashion_mnist", "abhishek/autotrain_fashion_mnist_vit_base"
        extractor = ViTFeatureExtractor(do_resize=False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        #extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=False)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    X_train, base_X_test = X_train.to(args.device), base_X_test.to(args.device)
    
    for alpha in alphas:
        scores = []
        for n_comp in n_components:
            pca = PCAAlpha(n_components=n_comp, alpha=alpha, random_state=SEED)
            pca.fit(X_train)
            X_test_ = pca.approximate(base_X_test)
            loss = (X_test_ - base_X_test).abs().mean().cpu()
            scores.append(loss)
            print("n_comp:", n_comp, "  alpha:", alpha, "  MAE:", loss)

        plt.plot(n_components, scores, label="alpha: " + str(alpha))
        

    plt.legend(loc="best")
    plt.xlabel("Number of components")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(path + plot_name, dpi=250)