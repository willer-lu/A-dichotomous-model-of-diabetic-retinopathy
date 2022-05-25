import random
from torchvision.transforms import functional as F
from torchvision import transforms


class Compose(object):
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, image):
        for t in self.transformations:
            image= t(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)

        return image


class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image
    

    
def get_transform(train):
    """Compose transforms to be applied to the train and validation sets"""
    transform = []

    img_size = 512
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transform.append(transforms.Resize(int(img_size*1.15)))
    transform.append(transforms.CenterCrop(img_size))




    if train:
        transform.append(RandomHorizontalFlip(0.2))
        transform.append(transforms.RandomInvert())
        transform.append(transforms.RandomVerticalFlip(0.2))
        transform.append(transforms.GaussianBlur(5, 10))


    return Compose(transform)
