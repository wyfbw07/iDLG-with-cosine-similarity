import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image

def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def reconstruction_costs(dummy_dy_dx, original_dy_dx):
    # grad_diff = 0
    # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
    #     grad_diff += ((gx - gy) ** 2).sum()
    # return grad_diff
    total_costs = 0
    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
        pnorm = [0, 0]
        costs = 0
        costs -= (gy * gx).sum()
        pnorm[0] += gy.pow(2).sum()
        pnorm[1] += gx.pow(2).sum()
        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        # Accumulate final costs
        total_costs += costs
    return total_costs / len(dummy_dy_dx)


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)

    if '.DS_Store' in folders:
        folders.remove('.DS_Store')

    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def main():
    dataset = 'cifar100'
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/iDLG_%s'%dataset).replace('\\', '/')
    
    lr = 0.1
    num_dummy = 1
    Iteration = 300 # LBFGS
    # Iteration = 2000
    num_exp = 1

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=False)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment'%(idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['DLG', 'iDLG', 'iDLG_cos']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []
            # imidx_list = [29688]

            for imidx in range(num_dummy):
                # idx = idx_shuffle[imidx]
                idx = 44046
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)


            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG_cos' or method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[Iteration // 2.667, Iteration // 1.6,
                                                                         Iteration // 1.142], gamma=0.1)   # 3/8 5/8 7/8

            print('lr =', lr)
            for iters in range(Iteration):
                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                         grad_diff += ((gx - gy) ** 2).sum()
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                         grad_diff += ((gx - gy) ** 2).sum()
                    elif method == 'iDLG_cos':
                        # Use cosine similarity
                        dummy_loss = criterion(pred, label_pred)
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                        grad_diff = 0
                        grad_diff = reconstruction_costs(dummy_dy_dx,original_dy_dx)
                        grad_diff += 1e-4 * total_variation(dummy_data)
                    grad_diff.backward()
                    # if method == 'iDLG':
                    #     dummy_data.grad.sign_()
                    return grad_diff

                optimizer.step(closure)
                if method == 'iDLG':
                    scheduler.step()
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())

                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                print(current_time, ',', iters, ',', '%.8f, %.8f' %(current_loss, mses[-1]))
                # history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                # history_iters.append(iters)

                if iters % int(Iteration / 30) == 0:
                    # current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    # print(current_time, ',', iters, ',', '%.8f, %.8f' %(current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG_cos':
                            plt.savefig('%s/iDLG_cos_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()

                    if current_loss < 0.000001: # converge
                        break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses
            elif method == 'iDLG_cos':
                loss_iDLG_cos = losses
                label_iDLG_cos = label_pred.item()
                mse_iDLG_cos = mses

        print('imidx_list:', imidx_list)
        print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1], 'loss_iDLG_cos:', loss_iDLG_cos[-1])
        print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1], 'mse_iDLG_cos:', mse_iDLG_cos[-1])
        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG, 'lab_iDLG_cos:', label_iDLG_cos)
        print('----------------------\n\n')

if __name__ == '__main__':
    main()


