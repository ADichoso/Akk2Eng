import os
import model.lstm as lstm
import model.test as test


def evaluate_lstm(checkpoint_dir, test_loader, device):
    results = {}

    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])

    if not checkpoints:
        print(f"No .pth files found in {checkpoint_dir}")
        return

    for filename in checkpoints:
        path = os.path.join(checkpoint_dir, filename)
        print(f"Testing {filename}...")

        model = lstm.load_lstm_checkpoint(path, device)
        _, _, _, _, geomean = test.test_model(
            model, test_loader, device, 0, "l", "beam"
        )

        results[filename] = geomean
        print(f"  Geometric Mean: {geomean:.4f}")

    best = max(results, key=results.get)

    print(f"Best checkpoint : {best}")
    print(f"Geometric Mean  : {results[best]:.4f}")

    return best, results