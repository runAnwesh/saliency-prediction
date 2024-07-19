import datetime
import os
import torch
import argparse
from torch.utils.data import DataLoader
from join_mul_gan import UnetGAN
from join_dataset import JPairDataset
from dataset import PairDataset
from discriminator import Discriminator
from util import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('/path/to/pytorch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--load", default=None)
    parser.add_argument('--dataset', default='./Dataset/train/')
    parser.add_argument('--train_list', default='./Dataset/train/train.lst')
    parser.add_argument('--train_edge_list', default='./Dataset/train/train_edge.lst')
    parser.add_argument('--cuda', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning_rate.', default=0.0001, type=float)

    args = parser.parse_args()
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    pair_dataset = PairDataset(args.dataset)
    Jpair_dataset = JPairDataset(args.dataset, args.train_list, args.train_edge_list)
    load = args.load
    modelG = UnetGAN().cuda()
    modelD = Discriminator().cuda()
    now = datetime.datetime.now()
    start_iter = 0
    start_epo = 0
    if load is not None:
        start_iter = 0
        start_epo = 180
        model_dict = modelG.state_dict()
        pretrained_dict = torch.load(load, map_location=args.cuda)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        modelG.load_state_dict(model_dict)
    # Optimizer Setup
    learning_rate = args.learning_rate
    optG = torch.optim.Adam(modelG.parameters(), lr=learning_rate)
    optD = torch.optim.Adam(modelD.parameters(), lr=learning_rate)
    weight_save_dir = os.path.join('models', 'state_dict')
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)

    dataloader = DataLoader(pair_dataset, batch_size, shuffle=True, num_workers=0)
    iterate = start_iter
    for epo in range(0, 180):
        iterate = start_iter
        n_updates = 1
        for i, batch in enumerate(dataloader):
            batch_num = batch['image'].size()[0]
            img = to_variable(batch['image'], requires_grad=False)
            depth = to_variable(batch['depth'], requires_grad=False)
            GT = to_variable(batch['mask'], requires_grad=False)
            rgb_labels = to_variable(torch.FloatTensor(np.ones(batch_num, dtype=float)), requires_grad=False)
            dep_labels = to_variable(torch.FloatTensor(np.zeros(batch_num, dtype=float)), requires_grad=False)
            optG.zero_grad()
            optD.zero_grad()
            BCE_function = torch.nn.BCELoss()
            if n_updates % 2 == 1:
                # -----Train the Discriminator-----
                pred_rgb, pred_dep, _ = modelG([img, depth, GT], mode=1)
                d_out_rgb = modelD(torch.cat((img, pred_rgb), 1)).squeeze()
                d_out_dep = modelD(torch.cat((img, pred_dep), 1)).squeeze()
                D_loss = BCE_function(d_out_rgb, rgb_labels.squeeze()) + BCE_function(d_out_dep, dep_labels.squeeze())
                D_loss.backward()
                optD.step()
            else:
                # -----Train the Generator-----
                pred_rgb, pred_dep, loss = modelG([img, depth, GT], mode=1)
                d_out_rgb = modelD(torch.cat((img, pred_rgb), 1)).squeeze()
                d_out_dep = modelD(torch.cat((img, pred_dep), 1)).squeeze()
                G_d_loss = BCE_function(d_out_rgb, dep_labels.squeeze()) + BCE_function(d_out_dep, rgb_labels.squeeze())
                G_loss = loss + G_d_loss
                print("loss:{:.3f}".format(G_loss))
                G_loss.backward()
                optG.step()
            n_updates += 1
            iterate += batch_num

    dataloader = DataLoader(Jpair_dataset, batch_size, shuffle=True, num_workers=0)
    for epo in range(0, 10):
        iterate = start_iter
        n_updates = 1
        for i, batch in enumerate(dataloader):
            batch_num = batch['image'].size()[0]
            img, depth, GT = to_variable(batch['image']), to_variable(batch['depth']), to_variable(batch['mask']),
            edgeimg, edgeGT = to_variable(batch['edge']), to_variable(batch['edge_gt'])
            optG.zero_grad()
            final_pred, edge_loss = modelG([edgeimg, edgeGT], mode=0)
            _, _, _, loss = modelG([img, depth, GT], mode=2)
            G_loss = loss + edge_loss
            print("{}/{}\tloss:{:.3f}".format(epo, iterate, loss))
            G_loss.backward()
            optG.step()
            n_updates += 1
            iterate += batch_num
    torch.save(modelG.state_dict(), os.path.join(weight_save_dir, 'cmSalGAN.ckpt'))
