import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch._C import Generator

dataset_root = r'dataset\DATA-IMAGE-FOLDER'
batch_size = 128
n_worker = 4
lr = 0.001
temperature = 6
alpha = 0.1
seed = 111
generator = Generator()
generator.manual_seed(seed)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size=(224, 224)
# mean = (0.4915, 0.4823, 0.4468)
# std = (0.2470, 0.2435, 0.2616)
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

transformers = {
    'original': T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),

    'dataset1': T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(2),
        # T.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]),

    'dataset2': T.Compose([
        T.RandomHorizontalFlip(p=0.68),
        T.RandomErasing(inplace=True, scale=(0.01, 0.23)),
        # T.RandomAffine(translate=(0.05, 0.05), degrees=0),
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),

    'dataset3': T.Compose([
        T.Resize(input_size),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomRotation(15),
        # T.RandomAffine(translate=(0.08, 0.1), degrees=15),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),
    
    'light': A.Compose([
        A.RandomBrightnessContrast(p=0.5),    
        A.RandomGamma(p=0.5),    
        A.CLAHE(p=0.5),
        A.HorizontalFlip(p=0.9),
        A.Resize(*input_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
        ], p=1.0),
    
    'medium': A.Compose([
        A.CLAHE(p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
        A.HorizontalFlip(p=0.6),
        A.MedianBlur(p=0.3),
        A.Resize(*input_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
        ], p=1.0),
        
    'strong' : A.Compose([
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
        A.ChannelShuffle(p=0.4),
        A.HorizontalFlip(p=0.6),
        A.JpegCompression(p=0.5),
        A.Resize(*input_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
        ], p=1.0)
}
