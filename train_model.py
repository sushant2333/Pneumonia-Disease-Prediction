
from fastai.vision.all import *
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def evaluate_test_set(inf_learner: Learner, test_path: Path) -> None:
    classes = list(map(str, inf_learner.dls.vocab))
    if len(classes) == 0:
        raise RuntimeError("Learner has empty vocab; cannot evaluate.")

    files = get_image_files(test_path)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under {test_path}.")

    cm = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    total = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for f in files:
        true_label = f.parent.name
        img = PILImage.create(f)
        pred_class, pred_idx, pred_probs = inf_learner.predict(img)
        true_idx = classes.index(true_label)
        pred_idx_int = int(pred_idx)
        cm[true_idx, pred_idx_int] += 1
        total += 1
        if true_idx == pred_idx_int:
            correct += 1
        y_true.append(true_idx)
        y_pred.append(pred_idx_int)

    acc = (correct / total) if total > 0 else 0.0
    print(f"Test images: {total}")
    print(f"Test accuracy: {acc:.4f}")

    # Classification report (precision/recall/F1 per class)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(6,6), dpi=120)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, int(cm[i, j].item()), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png")


def main() -> None:
    # Configuration
    base_path = Path(r"D:\Projects\pneumonia disease prediction\chest_xray")
    train_path = base_path/"train"
    test_path = base_path/"test"

    # Set random seed for reproducibility
    set_seed(42, reproducible=True)

    # DataBlock and DataLoaders (num_workers=0 to avoid Windows multiprocessing issues)
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(256),
        batch_tfms=[
            *aug_transforms(do_flip=True, flip_vert=False, max_rotate=15, max_zoom=1.1,
                            max_warp=0.0, max_lighting=0.2, p_affine=0.75, p_lighting=0.75),
            Normalize.from_stats(*imagenet_stats)
        ]
    )

    dls = dblock.dataloaders(train_path, bs=32, num_workers=0)

    export_file = Path("export.pkl")
    if export_file.exists():
        print("Found export.pkl — skipping training and running full test evaluation...")
        inf_learner = load_learner(export_file)
        evaluate_test_set(inf_learner, test_path)
        return

    # Model: pretrained ResNet50 (use vision_learner per fastai v2 API)
    learn = vision_learner(dls, resnet50, metrics=accuracy)

    # Train
    learn.fine_tune(5)

    # Evaluate and plot confusion matrix (validation)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(6,6), dpi=120)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    # Save trained model for future use
    learn.export("export.pkl")

    # Inference example: load the saved model and predict on a single image
    inf_learner = load_learner("export.pkl")

    # Pick an example test image (first found). You can replace this with a specific filename.
    test_images = get_image_files(test_path)
    if len(test_images) == 0:
        raise FileNotFoundError(f"No images found under {test_path}.")

    example_img_path = test_images[0]
    pred_class, pred_idx, pred_probs = inf_learner.predict(PILImage.create(example_img_path))
    print(f"Image: {example_img_path}")
    print(f"Prediction: {pred_class}; Probabilities: {pred_probs}")

    # Full test-set evaluation
    evaluate_test_set(inf_learner, test_path)


if __name__ == "__main__":
    main()
