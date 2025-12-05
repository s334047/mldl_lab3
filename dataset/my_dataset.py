from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def get_transforms():
 return T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_tiny_imagenet_datasets():
    transform = get_transforms()
    dataset_train = ImageFolder(
       root='tiny-imagenet/tiny-imagenet-200/train',
       transform=transform
    )
    dataset_val = ImageFolder(
       root='tiny-imagenet/tiny-imagenet-200/val',
       transform=transform
    )
    return dataset_train, dataset_val