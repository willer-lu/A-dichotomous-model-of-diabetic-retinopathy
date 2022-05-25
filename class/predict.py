import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from model import swin_base_patch4_window12_384_in22k as create_model
from albumentations import CLAHE, Compose

model_weight_path = "./models/model.pth"


test_dir = "./test"



def augment(im):
    image = np.array(im)
    image = np.uint8(image)
    light = Compose([
        CLAHE(p=1),
    ], p=1)
    image = light(image=image)['image']
    return Image.fromarray(image)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 512
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size)
        ]
        )

    model = create_model(num_classes = 2).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    img_name_list = os.listdir(test_dir)
    ans = []
    for img_name in tqdm(img_name_list):
        img_num = img_name.split(".")[0]
        img_path = os.path.join(test_dir,img_name)

        img = Image.open(img_path)
        img = augment(img)
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        ans.append(int(predict_cla))


    np.save("./pre.npy", ans)