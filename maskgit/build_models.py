import torch
from networks import VQGAN, MaskGIT
from utils import print_num_params


def build_models(config,
                 g_config=None,
                 d_config=None,
                 t_config=None):
    phase = config['phase']
    model = VQGAN(channels=3,
                  conv_in_channels=g_config['conv_in_channels'],
                  embed_dim=g_config['embed_dim'],
                  n_embed=g_config['n_embed'],
                  channels_mult=g_config['channels_mult'],
                  rec_loss_weight=g_config['rec_loss_weight'],
                  gan_loss_weight=g_config['gan_loss_weight'],
                  lpips_loss_weight=g_config['lpips_loss_weight'],
                  latent_size=g_config['latent_size'],
                  use_disc=config['use_disc'],
                  cuda=config['cuda'],
                  phase=phase,
                  d_config=d_config
                  )
    device = 'cuda' if config['cuda'] else 'cpu'
    if config['load']:
        model.load_state_dict(torch.load(config['load'],
                                         map_location=torch.device(device)),
                              strict=False)
        print("vq_loaded")
    model.eval()

    print_num_params(model.encoder, "Encoder",config_path=config['config_path'])
    print_num_params(model.code_book, "CodeBook",config_path=config['config_path'])
    print_num_params(model.decoder, "Decoder",config_path=config['config_path'])
    if phase == 1:
        print_num_params(model.lpips_fn, "LPIPS",config_path=config['config_path'])

    if config['use_disc']:
        print_num_params(model.discriminator, "Discriminator",config_path=config['config_path'])
    if config['cuda']:
        model.cuda()

    if phase == 1:
        g_optim = torch.optim.Adam(list(model.encoder.parameters()) + \
                                   list(model.decoder.parameters()),
                                   lr=g_config['base_learning_rate'],
                                   betas=(0.5, 0.9))
    if config['use_disc'] and phase == 1:
        d_optim = torch.optim.Adam(model.discriminator.parameters(),
                                   lr=d_config['base_learning_rate'],
                                   betas=(0.5, 0.9))
    if phase == 1 and config['use_disc']:
        return model, g_optim, d_optim
    elif phase == 1:
        return model, g_optim

    if phase == 2 or phase == 3:
        maskgit = MaskGIT(n_tokens=t_config['n_tokens'],
                          n_pos=t_config['n_pos'],
                          embed_dim=t_config['embed_dim'],
                          nhead=t_config['nhead'],
                          hidden_dim=t_config['hidden_dim'],
                          n_layers=t_config['n_layers'],
                          n_steps=t_config['n_steps'])
        device = 'cuda' if config['cuda'] else 'cpu'
        if t_config['load']:
            maskgit.load_state_dict(torch.load(t_config['load'],
                                               map_location=torch.device(device)),
                                    strict=False)
            print("maskgit_loaded")
        maskgit.eval()
        print_num_params(maskgit, "MaskGIT",config_path=config['config_path'])
        if config['cuda']:
            maskgit.cuda()
        t_optim = torch.optim.Adam(maskgit.parameters(),
                                   lr=t_config['base_learning_rate'],
                                   betas=(0.9, 0.96))
        return maskgit, t_optim, model
