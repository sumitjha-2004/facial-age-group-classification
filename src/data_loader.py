from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
print(len(train_dataset))
print(train_dataset.classes)

print(len(test_dataset))
print(test_dataset.classes)