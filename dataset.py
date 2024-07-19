import os
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# class Resize(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, sample):
#         img, mask = sample['image'], sample['mask']
#         img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR),\
#                     mask.resize((self.size, self.size),resample=Image.BILINEAR)
#         return {'image': img, 'mask': mask}


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, depth, mask = sample['image'], sample['depth'], sample['mask']
        img, depth, mask = img.resize((256, 256), resample=Image.BILINEAR), \
                                  depth.resize((256, 256), resample=Image.BILINEAR), \
                                  mask.resize((256, 256), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        depth = depth.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))
        return {'image': img, 'depth': depth, 'mask': mask}


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, depth, mask = sample['image'], sample['depth'], sample['mask']
            img = self.flip(img)
            depth = self.flip(depth)
            mask = self.flip(mask)
            return {'image': img, 'depth': depth, 'mask': mask}
        else:
            return sample


class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img, depth, mask = sample['image'], sample['depth'], sample['mask']
        img, depth, mask = self.tensor(img), self.tensor(depth), self.tensor(mask)
        return {'image': img, 'depth': depth, 'mask': mask}


class PairDataset(data.Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True):
        self.root_dir = root_dir
        self.train = train
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'RGB')))
        self.mask_list = sorted(os.listdir(os.path.join(root_dir, 'GT')))
        self.transformRGB = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])
        self.root_dir = root_dir
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, 'RGB', self.image_list[item])
        mask_name = os.path.join(self.root_dir, 'GT', self.mask_list[item])
        depth_name = os.path.join(self.root_dir, 'depth', self.image_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        depth = Image.open(depth_name)
        depth = depth.convert('RGB')
        img = img.convert('RGB')
        # img_norm = self.transformRGB(img)
        mask = mask.convert('L')
        sample = {'image': img, 'depth': depth, 'mask': mask}
        sample = self.transform(sample)
        sample['image'] = self.transformRGB(sample['image'])
        return sample


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.image_rgb_list = sorted(os.listdir(os.path.join(root_dir, 'RGB')))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.transformRGB = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_rgb_list)

    def __getitem__(self, item):
        #img_name = '{}/{}'.format(self.root_dir, self.image_rgb_list[item])
        img_name = self.image_rgb_list[item]
        img_path_rgb = os.path.join(self.root_dir, 'RGB', self.image_rgb_list[item])
        img_path_dep = os.path.join(self.root_dir, 'depth', self.image_rgb_list[item])
        img = Image.open(img_path_rgb)
        imgsize = img.size
        img_dep = Image.open(img_path_dep)
        sample_rgb = img.convert('RGB')
        # sample_rgb = self.transformRGB(sample_rgb)
        sample_dep = img_dep.convert('RGB')
        sample_rgb = self.transform(sample_rgb)
        sample_rgb = self.transformRGB(sample_rgb)
        # temppath = './Norm_img/'
        # os.makedirs(temppath, exist_ok=True)
        # torchvision.utils.save_image(sample_rgb, os.path.join(temppath, img_name))
        sample_dep = self.transform(sample_dep)
        return sample_rgb, sample_dep, img_name, imgsize