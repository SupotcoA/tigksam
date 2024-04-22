import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.decomposition import PCA


def cf2cl(tensor):
    return torch.permute(tensor, [0, 2, 3, 1])


def cl2cf(tensor):
    return torch.permute(tensor, [0, 3, 1, 2])


@torch.no_grad()
def print_num_params(model, name, config_path, **kwargs):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    with open(config_path, 'a') as f:
        f.write(f"{name} parameters: {num_params}\n")
    print(f"{name} parameters: {num_params}")


@torch.no_grad()
def save_phase1(model,
                phase,
                epoch,
                ver,
                outcome_root,
                latent_size,
                train_dataloader,
                test_dataloader,
                cuda,
                **kwargs
                ):
    assert phase==1
    torch.save(model.state_dict(),
               os.path.join(outcome_root,f"{ver}/vq{epoch}.pth"))
    n_pos=latent_size[0]*latent_size[1]
    model.eval()
    acc_encodings = []
    for _, batch_data in enumerate(train_dataloader):
        if cuda:
            batch_data = batch_data.cuda()
        encodings = model.encode(batch_data)
        acc_encodings.append(np.reshape(encodings.cpu().numpy().astype('uint16'),
                                        [-1,n_pos]))

    for _, batch_data in enumerate(test_dataloader):
        if cuda:
            batch_data = batch_data.cuda()
        encodings = model.encode(batch_data)
        acc_encodings.append(np.reshape(encodings.cpu().numpy().astype('uint16'),
                                        [-1,n_pos]))
    acc_encodings=np.vstack(acc_encodings)
    np.save(os.path.join(outcome_root, f"{ver}/{epoch}_enc.npy"),acc_encodings)

@torch.no_grad()
def save_phase2(maskgit,
                phase,
                ver,
                epoch,
                outcome_root,
                **kwargs):
    assert phase==2
    maskgit.eval()
    torch.save(maskgit.state_dict(),
               os.path.join(outcome_root,f"{ver}/maskgit{epoch}.pth"))

@torch.no_grad()
def vis_img(x, y, name,
            image_size,
            ver,
            outcome_root,
            **kwargs):
    h, w = image_size
    fp = os.path.join(outcome_root, f"{ver}/{name}.png")
    x = (cf2cl(x.detach().cpu()).numpy()[:8] + 1) / 2
    y = (cf2cl(y.detach().cpu()).numpy()[:8] + 1) / 2
    arr = np.zeros((4 * h, 4 * w, 3))
    for i in [0, 1]:
        for j in range(4):
            arr[2 * i * h:(2 * i + 1) * h, \
            j * w:(j + 1) * w] = x[4 * i + j, :, :, :]
    for i in [0, 1]:
        for j in range(4):
            arr[(2 * i + 1) * h:(2 * i + 2) * h, \
            j * w:(j + 1) * w] = y[4 * i + j, :, :, :]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(fp, arr)


def show_probs_rank(avg_probs,outcome_root,ver):
    avg_probs = np.sort(avg_probs)
    plt.bar(np.arange(avg_probs.shape[0]), avg_probs)
    plt.savefig(os.path.join(outcome_root, f"{ver}/Dis.png"), dpi=80)
    plt.clf()


def vis_pca(model,
            maskgit=None,
            config=None):
    if config['phase']==1:
        X = model.code_book.embed.weight.detach().cpu().numpy()
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        plt.scatter(X[:, 0], X[:, 1])
        plt.savefig(os.path.join(config['outcome_root'], f"{config['ver']}/PCA.png"),dpi=80)
        plt.clf()
    elif config['phase']==2 or config['phase']==3:
        def f(k):
            if 0<=k<=3: # 0 3
                return k
            elif 4<=k<=7: # 5 11
                return 2*k-3
            elif 8<=k<=11:  # 14 23
                return 3*k-10
            else:
                return 0
        X = maskgit.pos_embed.detach().cpu().numpy()
        pca = PCA(n_components=24)
        X = pca.fit_transform(X)
        fig, axs = plt.subplots(3, 4)
        for i,ax in enumerate(axs.flat):
            k=f(i)
            ax.imshow(np.reshape(X[:,k],config['latent_size']))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(str(k+1))
        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        plt.savefig(os.path.join(config['outcome_root'], f"{config['ver']}/PosEmbed.png"),dpi=80)
        plt.clf()


def make_archive(zipfile_path, dir_path, config):
    shutil.make_archive(os.path.join(zipfile_path,config['ver']),
                        "zip", os.path.join(dir_path,config['ver']))