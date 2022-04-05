import torch
import torchvision.transforms as T
from torch._C import Generator

dataset_root = 'dataset'
batch_size = 32
n_worker = 4
lr = 0.001
temperature = 6
alpha = 0.1

generator = Generator()
generator.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size=(224, 224)
mean = (0.4915, 0.4823, 0.4468)
std = (0.2470, 0.2435, 0.2616)

transformer = {
    'original': T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),

    'dataset1': T.Compose([
        T.Resize(input_size),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(5),
        T.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]),

    'dataset2': T.Compose([
        T.Resize(input_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.RandomAffine(translate=(0.05, 0.05), degrees=0),
        T.ToTensor(),
        T.RandomErasing(inplace=True, scale=(0.01, 0.23)),
        T.Normalize(mean, std)
    ]),

    'dataset3': T.Compose([
        T.Resize(input_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomAffine(translate=(0.08, 0.1), degrees=15),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
}
