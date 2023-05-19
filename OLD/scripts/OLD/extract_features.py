import argparse
import torch
from tqdm import tqdm
import os


@torch.inference_mode()
def extract_features(loader, model, device):
    from src.utils.metrics import OutputCollector

    model.eval()
    collector = OutputCollector()
    feats = []
    for batch in tqdm(loader, desc="Extracting features"):
        X, y, metadata = batch
        X = X.to(device)
        features = model(X).cpu()
        feats.append(features)
        collector.collect_batch({"y": y, **metadata})
    feats = torch.cat(feats)
    return feats, collector.compute(out_fmt="dataframe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet10_feature_extractor")
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="exact_patches_sl_all_centers_balanced_ndl",
    )
    parser.add_argument("--target_dir", type=str, required=True)

    args = parser.parse_args()

    from src.data.registry import create_dataset

    train_set = create_dataset(args.dataset_name, split="train")
    val_set = create_dataset(args.dataset_name, split="val")
    test_set = create_dataset(args.dataset_name, split="test")
    train_set.patch_transform = val_set.patch_transform

    from src.modeling.registry import create_model

    model = create_model(args.model_name)
    model.load_state_dict(torch.load(args.model_weights))

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_feats, train_meta = extract_features(train_loader, model, device)
    val_feats, val_meta = extract_features(val_loader, model, device)
    test_feats, test_meta = extract_features(test_loader, model, device)

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    torch.save(train_feats, args.target_dir + "/train_feats.pt")
    train_meta.to_csv(args.target_dir + "/train_meta.csv")
    torch.save(val_feats, args.target_dir + "/val_feats.pt")
    val_meta.to_csv(args.target_dir + "/val_meta.csv")
    torch.save(test_feats, args.target_dir + "/test_feats.pt")
    test_meta.to_csv(args.target_dir + "/test_meta.csv")


if __name__ == "__main__":
    main()
