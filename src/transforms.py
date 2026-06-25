from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir = "/content/drive/MyDrive/AI Age Detection Project/Split_DataSet/train"
test_dir = "/content/drive/MyDrive/AI Age Detection Project/Split_DataSet/test"

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

print("Train Images:", len(train_dataset))
print("Test Images:", len(test_dataset))
print("Classes:", train_dataset.classes)