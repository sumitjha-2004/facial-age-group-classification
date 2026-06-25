from tqdm import tqdm

model.eval()

correct = 0
total = 0

with torch.no_grad():

    test_bar = tqdm(
        test_loader,
        desc="Testing",
        leave=True
    )

    for images, labels in test_bar:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        current_acc = 100 * correct / total

        test_bar.set_postfix({
            "Accuracy": f"{current_acc:.2f}%"
        })

accuracy = 100 * correct / total

print(f"\nFinal Test Accuracy: {accuracy:.2f}%")