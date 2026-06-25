from tqdm import tqdm

num_epochs = 10

for epoch in range(num_epochs):

    # Training
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs}",
        leave=True
    )

    for images, labels in train_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        train_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Train Acc": f"{train_acc:.2f}%"
        })

    print(
        f"\nEpoch [{epoch+1}/{num_epochs}] "
        f"| Avg Loss: {running_loss/len(train_loader):.4f} "
        f"| Train Acc: {train_acc:.2f}%"
    )