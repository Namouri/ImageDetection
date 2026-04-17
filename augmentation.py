from torchvision import transforms

# Filen innehåller två funktioner som ni importerar i varje träningsfil:
# get_train_transform — augmentering + förbehandling för träningsdatan.
# get_val_transform — bara förbehandling utan augmentering för validering och test.
# Sedan i varje träningsfil använder ni dem så här:
#########################################################################
# pythonfrom augmentation import get_train_transform, get_val_transform

# train_transform = get_train_transform()
# val_transform = get_val_transform()

##########################################################################

def get_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])