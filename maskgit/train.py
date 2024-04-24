import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

# from networks import *
from utils import *


class EMALogger:
    def __init__(self, decay=0.9):
        self.decay = decay
        self.val = 0

    def update(self, val):
        self.val = self.decay * self.val + val * (1 - self.decay)


def train_step(model,
               epoch,
               train_dataloader,
               test_dataloader,
               d_optim,
               g_optim,
               config):
    model.train()
    n_batch = 0
    acc_rec_loss = 0
    acc_cb_loss = 0
    acc_lpips_loss = 0
    acc_gan_loss = 0
    acc_gan_acc = 0
    acc_encodings_sum = 0
    acc_d_loss = 0
    acc_d_acc_r = 0
    acc_d_acc_f = 0
    for batch_ind, batch_data in enumerate(train_dataloader):
        n_batch += 1
        if config['cuda']:
            batch_data = batch_data.cuda()

        d_optim.zero_grad()
        rec_out = model.reconstruct(batch_data)
        disc_out = model.discriminate(fake_x=rec_out['recx'].detach(),
                                      real_x=batch_data, on_train=True)
        d_loss = model.calculate_d_loss(disc_out['disc_out_fake'],
                                        disc_out['disc_out_real'])
        d_loss['tot_loss'].backward()
        d_optim.step()

        g_optim.zero_grad()
        disc_out = model.discriminate(fake_x=rec_out['recx'], on_train=True)
        g_loss = model.calculate_g_loss(x=batch_data,
                                        recx=rec_out['recx'],
                                        codebook_loss=rec_out['codebook_loss'],
                                        disc_out_fake=disc_out['disc_out_fake'])

        g_loss['tot_loss'].backward()
        g_optim.step()

        # d_loss_logger.update(d_loss['disc_loss'].detach().cpu().item())

        # model.update_gan_loss_weight(epoch=epoch,d_loss_val=d_loss_logger.val)
        # model.update_disc_loss_weight(epoch=epoch,d_loss_val=d_loss_logger.val)

        acc_rec_loss += g_loss['rec_loss'].detach()
        acc_cb_loss += g_loss['codebook_loss'].detach()
        acc_lpips_loss += g_loss['lpips_loss'].detach()
        acc_gan_loss += g_loss['gan_loss'].detach()
        acc_gan_acc += g_loss['gan_accuracy'].detach()
        acc_encodings_sum += rec_out['encodings_sum'].detach().cpu()
        acc_d_loss += d_loss['disc_loss'].detach()
        acc_d_acc_r += d_loss['disc_accuracy_real'].detach()
        acc_d_acc_f += d_loss['disc_accuracy_fake'].detach()

    avg_probs = acc_encodings_sum / torch.sum(acc_encodings_sum)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    if epoch % 1 == 0:
        with torch.no_grad():
            vis_img(batch_data, rec_out['recx'], f"train {epoch}",
                    config['image_size'],config['ver'],config['outcome_root'])
    if epoch % config['save_every_n_epoch'] == 0:
        save_phase1(config['phase'],
                    epoch,
                    config['ver'],
                    config['outcome_root'],
                    config['latent_size'],
                    train_dataloader,
                    test_dataloader,
                    config['cuda'],)
    if epoch % 3 == 1 and epoch >= 3:
        with torch.no_grad():
            model.code_book.embed.reinit(avg_probs.cuda())
    info = f'Train Epoch: {epoch}.\n' + \
           f'rec_loss: {acc_rec_loss / n_batch:.4f}; ' + \
           f'codebook_loss: {acc_cb_loss / n_batch:.4f}; ' + \
           f'lpips_loss: {acc_lpips_loss / n_batch:.4f}; ' + \
           f'gan_loss: {acc_gan_loss / n_batch:.4f}; ' + \
           f'gan_accuray: {acc_gan_acc / n_batch:.4f}; ' + \
           f'perplexity: {perplexity:.4f}; ' + \
           f'disc_loss: {acc_d_loss / n_batch:.4f}; ' + \
           f'disc_accuracy: {acc_d_acc_r / n_batch:.4f}/{acc_d_acc_f / n_batch:.4f};\n'
    with open(config['config_path'], 'a') as f:
        f.write(info)

    print(info)


@torch.no_grad()
def val_step(model,
             epoch,
             test_dataloader,
             config):
    model.eval()

    n_batch = 0
    acc_rec_loss = 0
    acc_cb_loss = 0
    acc_lpips_loss = 0
    acc_gan_loss = 0
    acc_gan_acc = 0
    acc_encodings_sum = 0
    acc_d_loss = 0
    acc_d_acc_r = 0
    acc_d_acc_f = 0

    for batch_ind, batch_data in enumerate(test_dataloader):
        n_batch += 1
        if config['cuda']:
            batch_data = batch_data.cuda()
        rec_out = model.reconstruct(batch_data)
        disc_out = model.discriminate(fake_x=rec_out['recx'],
                                      real_x=batch_data, on_train=False)
        d_loss = model.calculate_d_loss(disc_out['disc_out_fake'],
                                        disc_out['disc_out_real'])
        g_loss = model.calculate_g_loss(x=batch_data,
                                        recx=rec_out['recx'],
                                        codebook_loss=rec_out['codebook_loss'],
                                        disc_out_fake=disc_out['disc_out_fake'])
        acc_rec_loss += g_loss['rec_loss'].detach()
        acc_cb_loss += g_loss['codebook_loss'].detach()
        acc_lpips_loss += g_loss['lpips_loss'].detach()
        acc_gan_loss += g_loss['gan_loss'].detach()
        acc_gan_acc += g_loss['gan_accuracy'].detach()
        acc_encodings_sum += rec_out['encodings_sum'].detach().cpu()
        acc_d_loss += d_loss['disc_loss'].detach()
        acc_d_acc_r += d_loss['disc_accuracy_real'].detach()
        acc_d_acc_f += d_loss['disc_accuracy_fake'].detach()
    avg_probs = acc_encodings_sum / torch.sum(acc_encodings_sum)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    if epoch % 1 == 0:
        vis_img(batch_data, rec_out['recx'], f"test {epoch}",
                    config['image_size'],config['ver'],config['outcome_root'])
    info = f'Test Epoch: {epoch}.\n' + \
           f'rec_loss: {acc_rec_loss / n_batch:.4f}; ' + \
           f'codebook_loss: {acc_cb_loss / n_batch:.4f}; ' + \
           f'lpips_loss: {acc_lpips_loss / n_batch:.4f}; ' + \
           f'gan_loss: {acc_gan_loss / n_batch:.4f}; ' + \
           f'gan_accuray: {acc_gan_acc / n_batch:.4f}; ' + \
           f'perplexity: {perplexity:.4f}; ' + \
           f'disc_loss: {acc_d_loss / n_batch:.4f}; ' + \
           f'disc_accuracy: {acc_d_acc_r / n_batch:.4f}/{acc_d_acc_f / n_batch:.4f}\n'
    with open(config['config_path'], 'a') as f:
        f.write(info)
    if epoch == config['max_epochs']:
        show_probs_rank(avg_probs.numpy(),config['outcome_root'],config['ver'])
    print(info)


def train_step_vae(model,
                   epoch,
                   train_dataloader,
                   test_dataloader,
                   g_optim,
                   config):
    model.train()
    n_batch = 0
    acc_rec_loss = 0
    acc_cb_loss = 0
    acc_lpips_loss = 0
    acc_encodings_sum = 0
    for batch_ind, batch_data in enumerate(train_dataloader):
        n_batch += 1
        if config['cuda']:
            batch_data = batch_data.cuda()

        rec_out = model.reconstruct(batch_data)
        g_optim.zero_grad()
        g_loss = model.calculate_g_loss(x=batch_data,
                                        recx=rec_out['recx'],
                                        codebook_loss=rec_out['codebook_loss'],
                                        disc_out_fake=None)

        g_loss['tot_loss'].backward()
        g_optim.step()

        acc_rec_loss += g_loss['rec_loss'].detach()
        acc_cb_loss += g_loss['codebook_loss'].detach()
        acc_lpips_loss += g_loss['lpips_loss'].detach()
        acc_encodings_sum += rec_out['encodings_sum'].detach().cpu()

    avg_probs = acc_encodings_sum / torch.sum(acc_encodings_sum)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    if epoch % 1 == 0:
        with torch.no_grad():
            vis_img(batch_data, rec_out['recx'], f"train {epoch}",
                    config['image_size'],config['ver'],config['outcome_root'])
    if epoch % config['save_every_n_epoch'] == 0:
        save_phase1(config['phase'],
                    epoch,
                    config['ver'],
                    config['outcome_root'],
                    config['latent_size'],
                    train_dataloader,
                    test_dataloader,
                    config['cuda'], )
    if epoch % 3 == 1 and epoch >= 3:
        with torch.no_grad():
            model.code_book.embed.reinit(avg_probs.cuda())
    info = f'Train Epoch: {epoch}.\n' + \
           f'rec_loss: {acc_rec_loss / n_batch:.4f}; ' + \
           f'codebook_loss: {acc_cb_loss / n_batch:.4f}; ' + \
           f'lpips_loss: {acc_lpips_loss / n_batch:.4f}; ' + \
           f'perplexity: {perplexity:.4f}\n'

    with open(config['config_path'], 'a') as f:
        f.write(info)

    print(info)


@torch.no_grad()
def val_step_vae(model,
                 epoch,
                 test_dataloader,
                 config):
    model.eval()

    n_batch = 0
    acc_rec_loss = 0
    acc_cb_loss = 0
    acc_lpips_loss = 0
    acc_encodings_sum = 0

    for batch_ind, batch_data in enumerate(test_dataloader):
        n_batch += 1
        if config['cuda']:
            batch_data = batch_data.cuda()
        rec_out = model.reconstruct(batch_data)
        g_loss = model.calculate_g_loss(x=batch_data,
                                        recx=rec_out['recx'],
                                        codebook_loss=rec_out['codebook_loss'],
                                        disc_out_fake=None)
        acc_rec_loss += g_loss['rec_loss'].detach()
        acc_cb_loss += g_loss['codebook_loss'].detach()
        acc_lpips_loss += g_loss['lpips_loss'].detach()
        acc_encodings_sum += rec_out['encodings_sum'].detach().cpu()
    avg_probs = acc_encodings_sum / torch.sum(acc_encodings_sum)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    if epoch % 1 == 0:
        vis_img(batch_data, rec_out['recx'], f"test {epoch}",
                    config['image_size'],config['ver'],config['outcome_root'])
    info = f'Test Epoch: {epoch}.\n' + \
           f'rec_loss: {acc_rec_loss / n_batch:.4f}; ' + \
           f'codebook_loss: {acc_cb_loss / n_batch:.4f}; ' + \
           f'lpips_loss: {acc_lpips_loss / n_batch:.4f}; ' + \
           f'perplexity: {perplexity:.4f}\n'
    with open(config['config_path'], 'a') as f:
        f.write(info)
    if epoch == config['max_epochs']:  # to be removed!
        show_probs_rank(avg_probs.numpy(), config['outcome_root'],config['ver'])
    print(info)


def train_step_maskgit(maskgit,
                       epoch,
                       train_dataloader,
                       t_optim,
                       config,):
    maskgit.train()
    n_batch = 0
    acc_ce_loss = 0
    acc_raw_perplexity = 0

    for batch_ind, [batch_data] in enumerate(train_dataloader):
        n_batch += 1
        batch_data = batch_data.int()
        if config['cuda']:
            batch_data = batch_data.cuda()

        t_out = maskgit.train_val_step(batch_data)
        t_optim.zero_grad()
        t_loss = maskgit.calculate_loss(x=batch_data,
                                        logits=t_out['logits'],
                                        mask=t_out['mask'])

        t_loss['tot_loss'].backward()
        t_optim.step()

        acc_ce_loss += t_loss['tot_loss'].detach()
        acc_raw_perplexity += t_loss['raw_perplexity'].detach()

    perplexity = torch.exp(-acc_raw_perplexity / n_batch)
    if epoch % config['save_every_n_epoch'] == 0:
        save_phase2(epoch)
    info = f'Train Epoch: {epoch}.\n' + \
           f'ce_loss: {acc_ce_loss / n_batch:.4f}; ' + \
           f'perplexity: {perplexity:.2f}\n'
    with open(config['config_path'], 'a') as f:
        f.write(info)
    print(info)


@torch.no_grad()
def val_step_maskgit(maskgit,
                     epoch,
                     test_dataloader,
                     config):
    maskgit.eval()
    n_batch = 0
    acc_ce_loss = 0
    acc_raw_perplexity = 0

    for batch_ind, [batch_data] in enumerate(test_dataloader):
        n_batch += 1
        batch_data = batch_data.int()
        if config['cuda']:
            batch_data = batch_data.cuda()

        t_out = maskgit.train_val_step(batch_data)
        t_loss = maskgit.calculate_loss(x=batch_data,
                                        logits=t_out['logits'],
                                        mask=t_out['mask'])
        acc_ce_loss += t_loss['tot_loss']
        acc_raw_perplexity += t_loss['raw_perplexity']

    perplexity = torch.exp(-acc_raw_perplexity / n_batch)
    info = f'Test Epoch: {epoch}.\n' + \
           f'ce_loss: {acc_ce_loss / n_batch:.4f}; ' + \
           f'perplexity: {perplexity:.2f}\n'
    vis_maskgit_unconditional_generate(batch_size=8, name=f"m{epoch}")
    with open(config['config_path'], 'a') as f:
        f.write(info)
    print(info)


@torch.no_grad()
def vis_maskgit_unconditional_generate(maskgit,
                                       model,
                                       batch_size=8,
                                       name='1',
                                       t=None,
                                       config=None):
    if config['phase'] != 3:
        img_rows = []
        for _ in range(batch_size):
            ind_ls = maskgit.unconditional_generate(temperature=(1, 1))  # [n_step, 1, n_pos]
            if config['cuda']:
                ind = torch.vstack(ind_ls).cuda()
            else:
                ind = torch.vstack(ind_ls)
            imgs = cf2cl(model.decode(ind).cpu()).numpy()
            img_rows.append(np.hstack(imgs))
        output = np.vstack(img_rows)
        output = np.clip(output * 127.5 + 127.5, 0, 255).astype(np.uint8)
        fp = os.path.join(config['outcome_root'], f"{config['ver']}/{name}.png")
        cv2.imwrite(fp, output)

    steps_scheduler = lambda x: maskgit.n_steps - 4 + round(x / 16 * 8)
    temperature = t if t is not None else (1, 1)
    ind_ls = []
    for k in range(16):
        ind_ls.append(maskgit.unconditional_generate(temperature=temperature,
                                                     n_steps=steps_scheduler(k))[-1])
    if config['cuda']:
        ind = torch.vstack(ind_ls).cuda()
    else:
        ind = torch.vstack(ind_ls)
    imgs = model.decode(ind)
    vis_img(imgs[:8], imgs[8:], f"{name}_",
                    config['image_size'],config['ver'],config['outcome_root'])

    temperature = t if t is not None else (2.5, 1)
    ind_ls = []
    for k in range(16):
        ind_ls.append(maskgit.unconditional_generate(temperature=temperature,
                                                     n_steps=steps_scheduler(k))[-1])
    if config['cuda']:
        ind = torch.vstack(ind_ls).cuda()
    else:
        ind = torch.vstack(ind_ls)
    imgs = model.decode(ind)
    vis_img(imgs[:8], imgs[8:], f"{name}__",
                    config['image_size'],config['ver'],config['outcome_root'])


if __name__ == '__main__':
    pass
