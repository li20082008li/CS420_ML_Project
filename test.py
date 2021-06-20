import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1


def test(model, model_path, res_dir):
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    # preprocess
    test_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    # load data
    test_dataset = Dataset(imgs_dir='dataset/test_img', labels_dir='dataset/test_label', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print('Size of test set: ', len(test_dataset))

    # load model
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    i = 0
    model.eval()
    with torch.no_grad():
        for batch_num, data in tqdm(enumerate(test_loader)):
            img = data['img'].to(device)
            output = model(img)

            pred = to_numpy(classify_class(output))
            for j in range(pred.shape[0]):
                plt.imsave(res_dir + str(i) + '.png', pred[j].squeeze(), cmap='gray')
                i += 1

    print('Predict %d images completed.' % i)

