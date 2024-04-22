import os
import time
import torch
import numpy as np
from maskgit.data import build_dataset
from maskgit.build_models import build_models
from maskgit.train import (train_step, val_step,
                           train_step_vae, val_step_vae,
                           train_step_maskgit, val_step_maskgit,
                           vis_maskgit_unconditional_generate)
from maskgit.utils import (vis_pca, make_archive,
                           save_phase1, save_phase2)

torch.manual_seed(42)
np.random.seed(12)

phase = 2
# 1: training vae/gan & get encoding dataset
# 2: training transformer
# 3: product phase

config = {'ver': f'afhq_maskgit_0412_v02_phase{phase}',
          'description': 'train_again',
          'activation': 'relu',
          'batch_size': 32,
          'max_epochs': 2, 'train_test_split': 0.9,
          'save_every_n_epoch': 40,
          'dataset_size': 15228,
          'cuda': torch.cuda.is_available(),
          'image_size': (256, 256),  ###
          'load': False,  # '/kaggle/input/afhq-vqvae-0412-v01/vq12.pth',
          'save': True,
          'use_disc': False,
          'phase': phase
          }
# if phase!=1:
#    assert (not config['use_disc']) and config['lpips_loss_weight']<=0

config['Ds_ratio'] = 2 ** (len(config['channels_mult']) - 1)
config['latent_size'] = (round(config['image_size'][0] / config['Ds_ratio']),
                         round(config['image_size'][1] / config['Ds_ratio']))

g_config = {'channels': 3,
            'conv_in_channels': 64,
            'embed_dim': 256,
            'n_embed': 512,
            'channels_mult': (1, 1, 2, 2, 4),
            'rec_loss_weight': 1,
            'gan_loss_weight': 0.03,
            'lpips_loss_weight': 1,
            'latent_size': config['latent_size'],
            'use_disc': config['use_disc'],
            'base_learning_rate': 3.0e-4
            }

d_config = {'channels_mult': (1, 2, 4),
            'conv_in_channels': 64,
            'base_learning_rate': 8.0e-5,
            'disc_type': 'hinge',
            }

d_config['Ds_ratio'] = 2 ** (len(d_config['channels_mult']))
d_config['patch_size'] = (round(config['image_size'][0] / d_config['Ds_ratio']),
                          round(config['image_size'][1] / d_config['Ds_ratio']))

t_config = {'n_pos': 256,
            'n_tokens': 512,
            'embed_dim': 192,
            'nhead': 8,
            'hidden_dim': 768,
            'n_layers': 8,
            'n_steps': 10,
            'base_learning_rate': 2e-4,
            'min_learning_rate': 5e-5,
            'load': '/kaggle/input/maskgit-0412-v01/maskgit40.pth',
            # 'temperature':(3,1),
            }

assert t_config['n_pos'] == config['latent_size'][0] * config['latent_size'][1]

data_path_afhq = '/kaggle/input/afhq-512'
data_path_celeba = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
data_path_enc = '/kaggle/input/afhq-vqvae-0412-v01/12_enc.npy'
outcome_root = "/kaggle/working/vqvae"

config['data_path_afhq'] = data_path_afhq
config['data_path_celeba'] = data_path_celeba
config['data_path_enc'] = data_path_enc
config['outcome_root'] = outcome_root

if not os.path.exists(os.path.join(outcome_root, config['ver'])):
    os.makedirs(os.path.join(outcome_root, config['ver']))
config_path = os.path.join(outcome_root, f"{config['ver']}/config.txt")
config['config_path'] = config_path
with open(config_path, 'w') as f:
    f.write(str(config) + '\n')
    if config['use_disc']:
        f.write(str(d_config) + '\n')
    f.write(str(t_config) + '\n')

print(f"Cuda Availability: {config['cuda']}")

if phase == 1 and config['use_disc']:
    model, g_optim, d_optim = build_models(config=config,
                                           g_config=g_config,
                                           d_config=d_config,
                                           t_config=t_config)
elif phase == 1:
    model, g_optim = build_models(config=config,
                                  g_config=g_config,
                                  d_config=d_config,
                                  t_config=t_config)
elif phase == 2 or phase == 3:
    maskgit, t_optim, model = build_models(config=config,
                                           g_config=None,
                                           d_config=d_config,
                                           t_config=t_config)

if phase == 1 or phase == 2:
    train_dataloader, test_dataloader = build_dataset(phase=phase,
                                                      config=config,
                                                      data_path_afhq=config['data_path_afhq'],
                                                      data_path_celeba=config['data_path_celeba'],
                                                      data_path_enc=config['data_path_enc'])

if phase == 1 or phase == 2:
    t0 = time.time()
    t1 = time.time()
    # d_loss_logger = EMALogger(decay=0.9)
    for epoch in range(1, config['max_epochs'] + 1):
        t0 = t1
        if phase == 1:
            if config['use_disc']:
                train_step(model=model,
                           epoch=epoch,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           d_optim=d_optim,
                           g_optim=g_optim,
                           config=config
                           )
                val_step(model=model,
                         epoch=epoch,
                         test_dataloader=test_dataloader,
                         config=config
                         )
            else:
                train_step_vae(model=model,
                               epoch=epoch,
                               train_dataloader=train_dataloader,
                               test_dataloader=test_dataloader,
                               g_optim=g_optim,
                               config=config)
                val_step_vae(model=model,
                             epoch=epoch,
                             test_dataloader=test_dataloader,
                             config=config)
                # g_scheduler.step()
        elif phase == 2:
            train_step_maskgit(maskgit=maskgit,
                               epoch=epoch,
                               train_dataloader=train_dataloader,
                               t_optim=t_optim,
                               config=config)
            val_step_maskgit(maskgit=maskgit,
                             epoch=epoch,
                             test_dataloader=test_dataloader,
                             config=config)
            # t_scheduler.step()
        t1 = time.time()
        print(f"time: {t1 - t0:.2f}")

    with open(config_path, 'a') as f:
        f.write(f'time: {t1 - t0:.2f}\n')

    if config['save'] and config['max_epochs'] % config['save_every_n_epoch'] != 0:
        if phase == 1:
            save_phase1(config['max_epochs'])
        elif phase == 2:
            save_phase2(config['max_epochs'])

elif phase == 3:
    for i in np.linspace(0.5, 4.5, 9):
        vis_maskgit_unconditional_generate(batch_size=8, name=f"{i:.1f}_01", t=(i, 1))
        vis_maskgit_unconditional_generate(batch_size=8, name=f"{i:.1f}_02", t=(i, 1))

vis_pca(model, maskgit, config)
make_archive(zipfile_path="",
             dir_path="",
             config=config)
