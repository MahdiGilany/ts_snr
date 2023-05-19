import torch
from tqdm import tqdm
import sys
import os
import optuna

# append path one level up
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import src

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="exact_patches_sl_all_centers_balanced_ndl"
    )
    parser.add_argument("--model_name", type=str, default="resnet10_feature_extractor")
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--out_fname", type=str, default="eval_results.csv")
    parser.add_argument("--search", action="store_true", default=False)

    return parser.parse_args()


def extract_features(dataset, backbone):
    from src.utils.metrics import OutputCollector

    collector = OutputCollector()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4
    )
    for batch in tqdm(loader, desc="Extracting features"):
        X, y, metadata = batch
        with torch.no_grad():
            X = X.to(DEVICE)
            features = backbone(X)
            collector.collect_batch({"features": features, "y": y, **metadata})
    return collector.compute()


class LinEval:
    def __init__(self, train_X, train_Y, val_X, val_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.test_X = test_X
        self.test_Y = test_Y

    def __call__(self, trial: optuna.trial.Trial):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        l2_norm = trial.suggest_loguniform("l2_norm", 1e-5, 1e-1)
        clf = LogisticRegression(C=1 / l2_norm, max_iter=2000)

        clf.fit(self.train_X, self.train_Y)
        val_auroc = roc_auc_score(self.val_Y, clf.predict_proba(self.val_X)[:, 1])
        return val_auroc

    def test(self, l2_norm):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(C=1 / l2_norm, max_iter=2000)

        clf.fit(self.train_X, self.train_Y)

        y_hat = clf.predict_proba(self.test_X)[:, 1]

        test_auroc = roc_auc_score(self.test_Y, clf.predict_proba(self.test_X)[:, 1])
        return test_auroc, y_hat


def main():
    args = parse_args()

    from src.modeling.registry import create_model

    print(f"Loading model {args.model_name} from {args.model_weights}")
    model = create_model(args.model_name)
    model.load_state_dict(torch.load(args.model_weights))
    if hasattr(model, "backbone"):
        backbone = model.backbone
    else:
        backbone = model
    backbone.eval().to(DEVICE)
    from src.data.registry import create_dataset

    train_set = create_dataset(args.dataset_name, split="train")
    val_set = create_dataset(args.dataset_name, split="val")
    test_set = create_dataset(args.dataset_name, split="test")

    # extract features for entire dataset
    train_out = extract_features(train_set, backbone)
    val_out = extract_features(val_set, backbone)
    test_out = extract_features(test_set, backbone)

    objective = LinEval(
        train_out["features"],
        train_out["y"],
        val_out["features"],
        val_out["y"],
        test_out["features"],
        test_out["y"],
    )

    if args.search:
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=20)
        l2_norm = study.best_params["l2_norm"]
    else:
        l2_norm = 1e-3
    auroc, y_hat = objective.test(l2_norm)
    print(f"Test AUROC: {auroc}")

    # make table with probs (y_hat) and labels (test_out["y"]) and core_specifier (test_out["core_specifier"])
    import pandas as pd

    df = pd.DataFrame(
        {
            "y_hat": y_hat,
            "y": test_out["y"],
            "core_specifier": test_out["core_specifier"],
        }
    )
    df.to_csv(args.out_fname, index=False)


if __name__ == "__main__":
    main()
