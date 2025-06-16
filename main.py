import os
import torch
import argparse
from torch.utils.data import DataLoader, random_split
from transformers import logging as transformers_logging
from utils.dataset import MultimodalDataset
from utils.models import MDRIOG
from utils.metrics import evaluate

transformers_logging.set_verbosity_error()

def get_model_key(use_start, use_end, use_image):
    keys = []
    if use_start: keys.append("start")
    if use_end: keys.append("end")
    if use_image: keys.append("image")
    return "_".join(keys) if keys else "none"

# Training Configurations
best_val_acc = 0
patience, patience_limit = 0, 50
n_class = 6



def train_model(start_dir, end_dir, image_dir, use_start, use_end, use_image, device, log_path=None):
    global best_val_acc, patience, patience_limit, n_class
    # Get model keys
    model_key = get_model_key(use_start, use_end, use_image)

    # Created ./logs
    os.makedirs("logs", exist_ok=True)
    if log_path is None:
        log_path = f"logs/ablation_{model_key}.log"
    logf = open(log_path, 'w')
    def logprint(msg):
        print(msg)
        logf.write(msg + '\n')

    # Load Dataset
    dataset = MultimodalDataset(start_dir, end_dir, image_dir, n_class=n_class)
    lengths = [int(0.6 * len(dataset)), int(0.2 * len(dataset))]
    lengths.append(len(dataset) - sum(lengths))
    train_set, val_set, test_set = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

    loader_train = DataLoader(train_set, batch_size=64, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=64)
    loader_test = DataLoader(test_set, batch_size=64)

    # Configure model, optimizer, loss function
    model = MDRIOG(use_start, use_end, use_image, n_class=n_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    # Cosine annealing scheduler with T_max as total epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=300)  # 300 = total number of epochs

    os.makedirs("checkpoints", exist_ok=True)
    best_path = f"checkpoints/best_model_{model_key}.pt"

    for epoch in range(1, 301):
        model.train()
        total_loss = 0

        for batch in loader_train:
            inputs = {}
            if use_start:
                inputs["start_text"] = batch["start"]["text_emb"].to(device)
                inputs["start_feat"] = batch["start"]["feature"].to(device)
            if use_end:
                inputs["end_text"] = batch["end"].to(device)
            if use_image:
                inputs["image_emb"] = batch["image"].to(device)

            labels = batch["label"].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, loader_val, device, n_class)
        # learning rate scheduler step
        # scheduler.step()
        logprint(f"[Epoch {epoch}] Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), best_path)
            logprint("✅ Best model updated")
        else:
            patience += 1
            if patience >= patience_limit:
                logprint("⏹️ Early stopping triggered")
                break

    # Final Evaluation
    model.load_state_dict(torch.load(best_path))
    logprint("\n📊 Evaluation")
    _ = evaluate(model, loader_test, device, n_class, detailed=True, logf=logf)
    logf.close()

def eval_model(start_dir, end_dir, image_dir, use_start, use_end, use_image, device):
    model_key = get_model_key(use_start, use_end, use_image)
    dataset = MultimodalDataset(start_dir, end_dir, image_dir, n_class=n_class)
    lengths = [int(0.6 * len(dataset)), int(0.2 * len(dataset))]
    lengths.append(len(dataset) - sum(lengths))
    _, _, test_set = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    loader_test = DataLoader(test_set, batch_size=64)

    model = MDRIOG(use_start, use_end, use_image, n_class=n_class).to(device)
    best_path = f"checkpoints/best_model_{model_key}.pt"
    assert os.path.exists(best_path), f"Model checkpoint not found at {best_path}"
    model.load_state_dict(torch.load(best_path))

    print(f"\n📊 Evaluation for {model_key}")
    _ = evaluate(model, loader_test, device, n_class, detailed=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_dir', type=str, default="./pt_data/koclip_pt_start")
    parser.add_argument('--end_dir', type=str, default="./pt_data/koclip_pt_end")
    parser.add_argument('--image_dir', type=str, default="./pt_data/koclip_pt_image")
    parser.add_argument('--use_start', action='store_true')
    parser.add_argument('--use_end', action='store_true')
    parser.add_argument('--use_image', action='store_true')
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true', help="Only evaluate the saved best model")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        eval_model(
            start_dir=args.start_dir,
            end_dir=args.end_dir,
            image_dir=args.image_dir,
            use_start=args.use_start,
            use_end=args.use_end,
            use_image=args.use_image,
            device=device
        )
    else:
        train_model(
            start_dir=args.start_dir,
            end_dir=args.end_dir,
            image_dir=args.image_dir,
            use_start=args.use_start,
            use_end=args.use_end,
            use_image=args.use_image,
            device=device,
            log_path=args.log_path
        )