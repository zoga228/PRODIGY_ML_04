from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEMO_CLASSES = ["palm", "fist", "thumbs_up", "l_shape", "ok"]


class SmallGestureCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int) -> None:
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.targets = [label for _, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


class SyntheticGestureDataset(Dataset):
    def __init__(self, samples_per_class: int, image_size: int) -> None:
        self.classes = DEMO_CLASSES
        self.image_size = image_size
        self.items = [
            (class_index, item_index)
            for class_index in range(len(self.classes))
            for item_index in range(samples_per_class)
        ]
        self.targets = [class_index for class_index, _ in self.items]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        class_index, item_index = self.items[index]
        image = draw_demo_gesture(self.classes[class_index], item_index, self.image_size)
        return self.transform(image), class_index


def draw_demo_gesture(label: str, index: int, size: int) -> Image.Image:
    rng = np.random.default_rng(index + 101 * DEMO_CLASSES.index(label))
    base = rng.integers(0, 30, size=(size, size, 3), dtype=np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    color = tuple(int(value) for value in rng.integers(185, 256, size=3))
    offset_x = int(rng.integers(-5, 6))
    offset_y = int(rng.integers(-5, 6))
    cx = size // 2 + offset_x
    cy = size // 2 + offset_y

    if label == "palm":
        draw.ellipse((cx - 20, cy - 8, cx + 20, cy + 30), fill=color)
        for finger in range(-2, 3):
            x = cx + finger * 10
            draw.rounded_rectangle((x - 4, cy - 42, x + 4, cy + 5), radius=4, fill=color)
    elif label == "fist":
        draw.rounded_rectangle((cx - 30, cy - 20, cx + 30, cy + 30), radius=16, fill=color)
        for finger in range(4):
            draw.ellipse((cx - 28 + finger * 15, cy - 30, cx - 12 + finger * 15, cy - 10), fill=color)
    elif label == "thumbs_up":
        draw.rounded_rectangle((cx - 14, cy - 8, cx + 18, cy + 35), radius=10, fill=color)
        draw.rounded_rectangle((cx - 8, cy - 45, cx + 8, cy + 2), radius=8, fill=color)
        draw.rectangle((cx + 12, cy + 2, cx + 38, cy + 18), fill=color)
    elif label == "l_shape":
        draw.rounded_rectangle((cx - 34, cy + 8, cx + 28, cy + 24), radius=8, fill=color)
        draw.rounded_rectangle((cx - 34, cy - 44, cx - 18, cy + 18), radius=8, fill=color)
    else:
        draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), outline=color, width=9)
        draw.rounded_rectangle((cx + 12, cy + 12, cx + 36, cy + 34), radius=6, fill=color)

    return image


def discover_samples(root: Path, limit_per_class: int | None) -> tuple[list[tuple[Path, int]], list[str]]:
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found at {root}. Download the Kaggle dataset or run with --demo."
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            grouped[path.parent.name].append(path)

    classes = sorted(class_name for class_name, files in grouped.items() if len(files) >= 2)
    if len(classes) < 2:
        raise ValueError("At least two gesture classes with images are required.")

    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        files = sorted(grouped[class_name])
        if limit_per_class:
            files = files[:limit_per_class]
        samples.extend((file, class_to_index[class_name]) for file in files)
    return samples, classes


def make_loaders(dataset: Dataset, labels: list[int], batch_size: int) -> tuple[DataLoader, DataLoader]:
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    dataset: Dataset,
    classes: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(dataset, list(dataset.targets), args.batch_size)
    model = SmallGestureCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_total,
                "train_accuracy": train_correct / train_total,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch}: train_acc={history[-1]['train_accuracy']:.3f} "
            f"val_acc={val_accuracy:.3f}"
        )

    save_outputs(model, classes, history, y_true, y_pred, args)
    return {
        "classes": classes,
        "epochs": args.epochs,
        "validation_accuracy": float(history[-1]["val_accuracy"]),
        "train_images": int(len(train_loader.dataset)),
        "validation_images": int(len(val_loader.dataset)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def save_outputs(
    model: nn.Module,
    classes: list[str],
    history: list[dict],
    y_true: list[int],
    y_pred: list[int],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "image_size": args.image_size,
        },
        args.output_dir / "hand_gesture_cnn.pt",
    )
    (args.output_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, zero_division=0), encoding="utf-8"
    )
    plot_history(history, args.output_dir / "training_history.png")
    plot_confusion_matrix(y_true, y_pred, classes, args.output_dir / "confusion_matrix.png")


def plot_history(history: list[dict], path: Path) -> None:
    frame = pd.DataFrame(history)
    plt.figure(figsize=(7, 5))
    plt.plot(frame["epoch"], frame["train_accuracy"], marker="o", label="train")
    plt.plot(frame["epoch"], frame["val_accuracy"], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Hand Gesture CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], classes: list[str], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(max(6, len(classes) * 0.65), max(5, len(classes) * 0.55)))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Gesture Recognition Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])
    image_size = int(checkpoint["image_size"])
    model = SmallGestureCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, image_size


def predict_image(model_path: Path, image_path: Path, device: torch.device) -> str:
    model, classes, image_size = load_model(model_path, device)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    index = int(probabilities.argmax().item())
    return f"{classes[index]} ({probabilities[index].item():.2%})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand gesture recognition with a compact CNN.")
    parser.add_argument("--data-root", type=Path, default=Path("data/leapGestRecog"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--demo", action="store_true", help="Use generated gestures for a quick smoke test.")
    parser.add_argument("--demo-samples-per-class", type=int, default=80)
    parser.add_argument("--predict-image", type=Path, help="Classify one image using a saved model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/hand_gesture_cnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.predict_image:
        print(f"Predicted gesture: {predict_image(args.model, args.predict_image, device)}")
        return

    if args.demo:
        print("Running in demo mode with generated gesture images.")
        dataset = SyntheticGestureDataset(args.demo_samples_per_class, args.image_size)
        classes = dataset.classes
    else:
        samples, classes = discover_samples(args.data_root, args.limit_per_class)
        dataset = ImagePathDataset(samples, args.image_size)

    metrics = train_model(dataset, classes, args, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
