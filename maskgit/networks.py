import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
try:
    import lpips
except:
    pass

from modules import *
from vq import *


class VQGAN(nn.Module):

    def __init__(self,
                 channels=3,
                 conv_in_channels=64,
                 embed_dim=256,
                 n_embed=512,
                 channels_mult=(1, 1, 2, 2, 4),
                 rec_loss_weight=1,
                 gan_loss_weight=1,
                 lpips_loss_weight=1,
                 latent_size=(16,16),
                 use_disc=False,
                 cuda=False,
                 phase=1,
                 d_config=None):

        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(in_channels=channels,
                               conv_in_channels=conv_in_channels,
                               out_channels=embed_dim,
                               channels_mult=channels_mult)
        self.code_book = CodeBook(embed_dim=embed_dim,
                                  n_embed=n_embed)
        self.decoder = Decoder(in_channels=embed_dim,
                               conv_in_channels=conv_in_channels,
                               out_channels=channels,
                               channels_mult=channels_mult)
        if use_disc and phase == 1:
            self.discriminator = Discriminator(in_channels=channels,
                                               conv_in_channels=d_config['conv_in_channels'],
                                               out_channels=1,
                                               channels_mult=d_config['channels_mult'])
        self.rec_loss_weight = rec_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.latent_size = latent_size
        self.disc_loss_weight = 1
        self.use_disc = use_disc and phase == 1
        if lpips_loss_weight > 0 and phase == 1:
            self.lpips_loss_weight = lpips_loss_weight
            self.lpips_fn = lpips.LPIPS(net='vgg')
            if cuda:
                self.lpips_fn.cuda()
        else:
            self.lpips_loss_weight = None

    @torch.no_grad()
    def update_gan_loss_weight(self, epoch, d_loss_val):
        return
        # if epoch<=3:
        #   self.gan_loss_weight = 0.15
        # elif 5<epoch<9:
        #    self.gan_loss_weight = 0.06*(epoch-4)
        # else:
        #   self.gan_loss_weight = 0.3
        # self.gan_loss_weight = self.gan_loss_weight * max(0.2,-d_loss_val)

    @torch.no_grad()
    def update_disc_loss_weight(self, epoch, d_loss_val):
        return
        # self.disc_loss_weight = 1 #max(0.5,(d_loss_val+1)/2)

    @torch.no_grad()
    def encode(self, x):
        z_e = self.encoder(x)
        codebook_output = self.code_book(z_e)
        return codebook_output['encodings']

    @torch.no_grad()
    def decode(self, ind):
        z_q = self.code_book.embed(ind).view([-1, self.latent_size[0],
                                              self.latent_size[1],
                                              self.embed_dim]).contiguous()
        z_q = torch.permute(z_q, [0, 3, 1, 2]).contiguous()
        recx = self.decoder(z_q)
        return recx

    @torch.no_grad()
    def calculate_adaptive_weight(self, nll_loss, g_loss):
        last_layer = self.decoder.conv_out.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4)
        # d_weight = d_weight * self.discriminator_weight
        return d_weight

    def reconstruct(self, x):
        z_e = self.encoder(x)
        codebook_output = self.code_book(z_e)
        z_q, codebook_loss = codebook_output['z_q'], codebook_output['loss']
        recx = self.decoder(z_q)
        recx = torch.clip(recx, -1, 1)

        return {'recx': recx,
                'codebook_loss': codebook_loss,
                'encodings_sum': codebook_output['encodings_sum'],
                'encodings': codebook_output['encodings'],
                }

    def discriminate(self, fake_x, real_x=None, on_train=True):
        if real_x is None:
            self.discriminator.eval()
            disc_out_fake = self.discriminator(fake_x)
            if on_train:
                self.discriminator.train()
            disc_out_real = None
        else:
            disc_out_fake = self.discriminator(fake_x)
            disc_out_real = self.discriminator(real_x)
        return {'disc_out_fake': disc_out_fake,
                'disc_out_real': disc_out_real}

    @staticmethod
    def calculate_bce(self, proba, target=1):
        proba = F.sigmoid(proba)
        if target == 1:
            return -torch.mean(torch.log(proba + 1e-2))
        elif target == 0:
            return -torch.mean(torch.log((1 - proba) + 1e-2))

    def calculate_g_loss(self, x, recx, codebook_loss, disc_out_fake=None):
        rec_loss = torch.mean(torch.abs(recx - x))  # + torch.abs(recx - x) * 0.05
        lpips_loss = self.lpips_fn.forward(recx, x).mean()
        if self.use_disc:
            gan_loss = F.relu_(1.0 - disc_out_fake).mean()  # self.calculate_bce(disc_out_fake, target=1)
            gan_acc = 1 - torch.mean(disc_out_fake)
            ada_gan_loss_weight = self.calculate_adaptive_weight(rec_loss+lpips_loss,
                                                                 gan_loss)
        else:
            gan_loss = 0
            gan_acc = 0
            ada_gan_loss_weight = 0

        tot_loss = rec_loss * self.rec_loss_weight + \
                   codebook_loss + \
                   lpips_loss * self.lpips_loss_weight + \
                   gan_loss * ada_gan_loss_weight * self.gan_loss_weight
        return {'rec_loss': rec_loss,
                'codebook_loss': codebook_loss,
                'lpips_loss': lpips_loss,
                'gan_loss': gan_loss,
                'gan_accuracy': gan_acc,
                'tot_loss': tot_loss
                }

    def calculate_d_loss(self, disc_out_fake, disc_out_real):
        #disc_loss = self.calculate_bce(disc_out_fake, target=0) + \
        #            self.calculate_bce(disc_out_real, target=1)
        disc_loss = torch.mean(F.relu_(1. - disc_out_real)) + \
                    torch.mean(F.relu_(1. + disc_out_fake))
        tot_loss = disc_loss * self.disc_loss_weight
        disc_acc_r = torch.mean(disc_out_real)
        disc_acc_f = 1 - torch.mean(disc_out_fake)
        return {'disc_loss': disc_loss,
                'tot_loss': tot_loss,
                'disc_accuracy_real': disc_acc_r,
                'disc_accuracy_fake': disc_acc_f,
                }


class MaskGIT(nn.Module):

    def __init__(self, n_tokens=512,
                 n_pos=143,
                 embed_dim=128,
                 nhead=8,
                 hidden_dim=1024,
                 n_layers=6,
                 n_steps=8,
                 **kwargs):
        super().__init__()

        self.n_steps = n_steps
        self.n_pos = n_pos
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens

        weight = torch.randn(n_tokens, embed_dim)
        self.token_embed = nn.Parameter(weight, requires_grad=True)
        init.trunc_normal_(self.token_embed, 0, 0.02)
        weight = torch.randn(n_pos, embed_dim)
        self.pos_embed = nn.Parameter(weight, requires_grad=True)
        init.trunc_normal_(self.pos_embed, 0, 0.02)

        self.mask_embed = nn.Parameter(torch.zeros([embed_dim, ]), requires_grad=True)

        self.embed_out = nn.ModuleList([nn.GELU(),
                                        nn.LayerNorm(embed_dim)])

        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                 nhead=nhead,
                                                                 dim_feedforward=hidden_dim,
                                                                 dropout=0.1,
                                                                 activation='relu',
                                                                 batch_first=True)
                                      for _ in range(n_layers)])

        self.proj_out = nn.ModuleList([nn.Linear(embed_dim, embed_dim),
                                       nn.GELU(),
                                       nn.LayerNorm(embed_dim)])
        self.bias = nn.Parameter(torch.zeros([n_pos, n_tokens]), requires_grad=True)
        self.debugger = None

    @torch.no_grad()
    def calculate_n_mask(self, x=None):
        n = torch.cos(x * 3.1415926535 / 2) * self.n_pos
        n = torch.round(n).int()
        return n.item()

    @torch.no_grad()
    def sample_mask(self):
        n = self.calculate_n_mask(x=torch.rand((1,)))
        mask = torch.full((self.n_pos,), False, dtype=torch.bool)
        r = torch.rand((self.n_pos,))
        _, selected_positions = torch.topk(r, k=n, dim=-1)  # (n_masked,)
        mask[selected_positions] = True
        return mask

    def embed(self, ind, mask=None):  # ind/mask [batch_size,n_pos] or [n_pos,]
        embedding = self.token_embed[ind]  # [batch_size,n_pos,embed_dim] or [n_pos,embed_dim]
        if mask is not None:
            embedding[mask] = self.mask_embed  # [n_masked,embed_dim] <- [embed_dim,]
        embedding = embedding + self.pos_embed  # [(batch_size,)n_pos,embed_dim]+[n_pos,embed_dim]
        return embedding

    def train_val_step(self, x):  # x[batch_size, n_pos]
        masks = []
        for b in range(x.shape[0]):
            masks.append(self.sample_mask())
        mask = torch.vstack(masks)
        embedding = self.embed(x, mask=mask)
        logits = self.forward(masked_embedding=embedding)
        return {'logits': logits, 'mask': mask}

    def calculate_loss(self, x, logits, mask):
        logits_ = logits[mask].view(-1, self.n_tokens).contiguous()
        x_ = x[mask].view(-1).contiguous().long()
        ce_loss = F.cross_entropy(logits_,
                                  target=x_,
                                  label_smoothing=0.1).mean()
        log_proba = F.log_softmax(logits_.detach(), dim=-1)
        raw_perplexity = torch.gather(log_proba, dim=1, index=x_.unsqueeze(-1)).mean()
        return {"tot_loss": ce_loss, "raw_perplexity": raw_perplexity}

    def forward(self, masked_embedding):
        h = masked_embedding
        for layer in self.embed_out:
            h = layer(h)
        for layer in self.encoder:
            h = layer(h)
        for layer in self.proj_out:
            h = layer(h)
        logits = torch.matmul(h, self.token_embed.T) + self.bias
        return logits

    @torch.no_grad()
    def unconditional_generate(self, temperature=(1, 1), n_steps=None):
        # assert batch_size == 1
        if n_steps is None:
            n_steps = self.n_steps
        self.eval()
        ind_ls = []
        current_ind = (torch.rand((self.n_pos,)) * (self.n_tokens - 1)).long()  # [n_pos,]
        n_masked = self.n_pos
        mask = torch.full((self.n_pos,), True, dtype=torch.bool)  # [n_pos,]
        for t in range(1, n_steps):
            embedding = self.embed(current_ind, mask=mask)  # [n_pos,embed_dim]
            logits = self.forward(masked_embedding=embedding.view(1, self.n_pos,  # [n_pos,n_tokens]
                                                                  self.embed_dim)).squeeze(0)
            masked_logits = logits.clone()[mask] / temperature[0]  # [n_masked,n_tokens]
            token_dis = torch.distributions.categorical.Categorical(logits=masked_logits)
            token_sample = token_dis.sample()  # [n_masked,]
            token_confidence = torch.gather(token_dis.probs,  # [n_masked,]
                                            dim=-1,
                                            index=token_sample.unsqueeze(-1)).squeeze(-1)

            sorted_confidence, _ = torch.sort(token_confidence,
                                              dim=-1, descending=True)
            n = self.calculate_n_mask(x=torch.tensor(t / n_steps,
                                                     dtype=torch.float32).view(1, ))
            dn = n_masked - n
            n_masked = n
            threshold_confidence = sorted_confidence[dn]  # [1,]
            confident_token_flag = token_confidence > threshold_confidence  # [n_masked,]
            current_ind[mask] = torch.where(confident_token_flag.cpu(),
                                            token_sample.cpu(),
                                            current_ind[mask])
            mask[mask.clone()] = ~confident_token_flag.cpu()

            assert torch.abs(torch.sum(mask) - n_masked).cpu().item() <= 1
            ind_ls.append(current_ind.clone().squeeze(0))
        embedding = self.embed(current_ind, mask=mask)  # [n_pos,embed_dim]
        logits = self.forward(masked_embedding=embedding.view(1, self.n_pos,  # [n_pos,n_tokens]
                                                              self.embed_dim).contiguous()).squeeze(0)
        masked_logits = logits.clone()[mask]  # [n_masked,n_token]
        token_dis = torch.distributions.categorical.Categorical(logits=masked_logits)
        token_sample = token_dis.sample()  # [n_masked,]
        current_ind[mask] = token_sample.cpu()
        ind_ls.append(current_ind.clone().squeeze(0))
        return ind_ls



