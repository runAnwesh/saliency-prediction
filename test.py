from join_mul_gan import UnetGAN
from join_dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import argparse
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


torch.set_printoptions(profile='full')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--model_dir", default='./models/cmSalGAN.ckpt')
    parser.add_argument('--cuda', help="cuda for cuda, cpu for cpu, default = cuda", default='cuda')    
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    args = parser.parse_args()

    print(args)
    print(os.getcwd())
    device = torch.device(args.cuda)
    state_dict = torch.load(args.model_dir, map_location=args.cuda)
    modelG = UnetGAN().to(device)
    modelG.load_state_dict(state_dict)
    modelG.eval()
    
    
    setPath='./Dataset/NJUD/test/'
    savePath='./result/salMaps/test_njud'
    custom_dataset = CustomDataset(setPath)
    dataloader = DataLoader(custom_dataset, 1, shuffle=False)
    os.makedirs(os.path.join(savePath), exist_ok=True)
    for i, batch in enumerate(dataloader):
        print(i)
        img_rgb, img_dep, img_name, size = batch
        img_name = img_name[0].split('.jpg')[0] + '.png'
        img_rgb = img_rgb.to(device)
        img_dep = img_dep.to(device)
        with torch.no_grad():
            pred = modelG([img_rgb, img_dep], mode=2)
        if savePath is not None:
            pred_r = F.interpolate(pred, [size[1], size[0]], mode='bilinear', align_corners=True)
            pred_r = np.squeeze(pred_r.cpu().data.numpy()) * 255
            cv2.imwrite(os.path.join(savePath, img_name), pred_r)
            
    setPath='./Dataset/NLPR/test/'
    savePath='./result/salMaps/test_nlpr'
    custom_dataset = CustomDataset(setPath)
    dataloader = DataLoader(custom_dataset, 1, shuffle=False)
    os.makedirs(os.path.join(savePath), exist_ok=True)
    for i, batch in enumerate(dataloader):
        print(i)
        img_rgb, img_dep, img_name, size = batch
        img_name = img_name[0].split('.jpg')[0] + '.png'
        img_rgb = img_rgb.to(device)
        img_dep = img_dep.to(device)
        with torch.no_grad():
            pred = modelG([img_rgb, img_dep], mode=2)
        if savePath is not None:
            pred_r = F.interpolate(pred, [size[1], size[0]], mode='bilinear', align_corners=True)
            pred_r = np.squeeze(pred_r.cpu().data.numpy()) * 255
            cv2.imwrite(os.path.join(savePath, img_name), pred_r)

    setPath='./Dataset/STEREO/test/'
    savePath='./result/salMaps/test_stereo'
    custom_dataset = CustomDataset(setPath)
    dataloader = DataLoader(custom_dataset, 1, shuffle=False)
    os.makedirs(os.path.join(savePath), exist_ok=True)
    for i, batch in enumerate(dataloader):
        print(i)
        img_rgb, img_dep, img_name, size = batch
        img_name = img_name[0].split('.jpg')[0] + '.png'
        img_rgb = img_rgb.to(device)
        img_dep = img_dep.to(device)
        with torch.no_grad():
            pred = modelG([img_rgb, img_dep], mode=2)
        if savePath is not None:
            pred_r = F.interpolate(pred, [size[1], size[0]], mode='bilinear', align_corners=True)
            pred_r = np.squeeze(pred_r.cpu().data.numpy()) * 255
            cv2.imwrite(os.path.join(savePath, img_name), pred_r)

