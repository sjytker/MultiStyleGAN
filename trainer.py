"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MultiStyle_Gen, MsImageDis
from utils import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class MultiStyle_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MultiStyle_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        # TODO: ------------------Core Change--------------
        # self.gen_sc = AdaINGen(hyperparameters['input_dim_sc'], hyperparameters['gen'])  # auto-encoder for domain sc
        # self.gen_dw = AdaINGen(hyperparameters['input_dim_dw'], hyperparameters['gen'])  # auto-encoder for domain dw
        # self.gen_sw = AdaINGen(hyperparameters['input_dim_sw'], hyperparameters['gen'])  # auto-encoder for domain sw
        self.gen = MultiStyle_Gen(hyperparameters['input_dim_sc'], hyperparameters['gen'])
        self.dis_sc = MsImageDis(hyperparameters['input_dim_sc'], hyperparameters['dis'])  # discriminator for domain sc
        self.dis_dw = MsImageDis(hyperparameters['input_dim_dw'], hyperparameters['dis'])  # discriminator for domain dw
        self.dis_sw = MsImageDis(hyperparameters['input_dim_sw'], hyperparameters['dis'])  # discriminator for domain sw
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        # TODO: ------------------Core Change--------------
        display_size = int(hyperparameters['display_size'])
        self.s_t_sc = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_sc = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_t_dw = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_dw = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_t_sw = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_sw = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_sc.parameters()) + list(self.dis_dw.parameters()) + list(self.dis_sw.parameters())
        gen_params = list(self.gen_sc.parameters()) + list(self.gen_dw.parameters()) + list(self.gen_sw.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_sc.apply(weights_init('gaussian'))
        self.dis_dw.apply(weights_init('gaussian'))

        self.logger = logging.getLogger(self.__class__.__name__)

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def gen_update(self, x_sc, x_dw, x_sw, hyperparameters):
        """
            Update generator.
        """
        self.gen_opt.zero_grad()
        s_sc_t = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_sc_p = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_dw_t = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_dw_p = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_sw_t = Variable(torch.randn(x_sw.size(0), self.style_dim, 1, 1).cuda())
        s_sw_p = Variable(torch.randn(x_sw.size(0), self.style_dim, 1, 1).cuda())

        # ------------------------encode-------------------------------------
        c_sc, s_sc_t_prime, s_sc_p_prime = self.gen_sc.encode(x_sc)
        c_dw, s_dw_t_prime, s_dw_p_prime = self.gen_dw.encode(x_dw)
        c_sw, s_sw_t_prime, s_sw_p_prime = self.gen_sw.encode(x_sw)

        # ------------------------decode (dw)-----------------------
        # recon
        x_dw1 = self.decode(
            c_dw,
            [s_dw_t_prime, s_dw_p_prime],
            self.gen_dw,
            [self.gen_dw, self.gen_dw]
        )

        x_dw2 = self.decode(
            c_dw, 
            [s_sw_t, s_dw_p_prime],
            self.gen_dw, 
            [self.gen_sw, self.gen_dw]
        )

        # -------------------------decode (sw)-------------------------
        # recon
        x_sw1 = self.decode(
            c_sw, 
            [s_sw_t_prime, s_sw_p_prime],
            self.gen_sw,
            [self.gen_sw, self.gen_sw]
        )

        x_sw2 = self.decode(
            c_sw, 
            [s_sw_t_prime, s_sc_p],
            self.gen_sw,
            [self.gen_sw, self.gen_sc]
        )

        x_sw3 = self.decode(
            c_sw, 
            [s_dw_t, s_sw_p_prime],
            self.gen_sw,
            [self.gen_dw, self.gen_sw]
        )       

        x_sw4 = self.decode(
            c_sw, 
            [s_dw_t, s_sc_p],
            self.gen_sw,
            [self.gen_dw, self.gen_sc]
        )

        # -------------------------decode (sc)-------------------------
        # recon
        x_sc1 = self.decode(
            c_sc, 
            [s_sc_t_prime, s_sc_p_prime],
            self.gen_sc,
            [self.gen_sc, self.gen_sc]
        )

        x_sc2 = self.decode(
            c_sc, 
            [s_sc_t_prime, s_sw_p],
            self.gen_sc,
            [self.gen_sc, self.gen_sw]
        )       

        x_sc3 = self.decode(
            c_dw, 
            [s_sc_t, s_sc_p],
            self.gen_dw,
            [self.gen_sc, self.gen_sc]
        )

        x_sc4 = self.decode(
            c_dw, 
            [s_sc_t, s_sw_p],
            self.gen_dw,
            [self.gen_sc, self.gen_sw]
        )

        x_sc5 = self.decode(
            c_sw, 
            [s_sc_t, s_sc_p],
            self.gen_sw,
            [self.gen_sc, self.gen_sc]
        )

        x_sc6 = self.decode(
            c_sw, 
            [s_sc_t, s_sw_p_prime],
            self.gen_sw,
            [self.gen_sc, self.gen_sw]
        )
        # --------------------------encode again------------------------------------
        # c_dw_recon, s_sc_t_recon, s_sc_p_recon = self.gen_sc.encode(x_dw2sc)
        # c_sc_recon, s_dw_t_recon, s_dw_p_recon = self.gen_dw.encode(x_sc2dw)
        c_sc_recon, c_dw_recon, c_sw_recon = [], [], []
        s_sc_t_recon, s_sc_t_prime_recon, s_sc_p_recon, s_sc_p_prime_recon = [], [], [], []
        s_dw_t_recon, s_dw_t_prime_recon, s_dw_p_recon, s_dw_p_prime_recon = [], [], [], []
        s_sw_t_recon, s_sw_t_prime_recon, s_sw_p_recon, s_sw_p_prime_recon = [], [], [], []

        c, s_t, s_p = self.

        # --------------------------decode again (if needed)------------------------

        # --------------------reconstruction loss----------------------------
        # self.loss_gen_recon_x_sc = self.recon_criterion(x_sc_recon, x_sc)
        # self.loss_gen_recon_x_dw = self.recon_criterion(x_dw_recon, x_dw)
        # self.loss_gen_recon_s_sc_t = self.recon_criterion(s_sc_t_recon, s_sc_t)
        # self.loss_gen_recon_s_sc_p = self.recon_criterion(s_sc_p_recon, s_sc_p)
        # self.loss_gen_recon_s_dw_t = self.recon_criterion(s_dw_t_recon, s_dw_t)
        # self.loss_gen_recon_s_dw_p = self.recon_criterion(s_dw_p_recon, s_dw_p)
        # self.loss_gen_recon_c_sc = self.recon_criterion(c_sc_recon, c_sc)
        # self.loss_gen_recon_c_dw = self.recon_criterion(c_dw_recon, c_dw)
        # self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # ---------------------------GAN loss------------------------------------
        # self.loss_gen_adv_sc = self.dis_sc.calc_gen_loss(x_dw2sc)
        # self.loss_gen_adv_dw = self.dis_dw.calc_gen_loss(x_sc2dw)

        # --------------------------domain-invariant perceptual loss------------------------
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # --------------------------total loss--------------------------------------------------
        # the last 4 items are not shown in paper.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_sc + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_dw + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_sc + \
                              hyperparameters['recon_s_t_w'] * self.loss_gen_recon_s_sc_t + \
                              hyperparameters['recon_s_p_w'] * self.loss_gen_recon_s_sc_p + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_sc + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_dw + \
                              hyperparameters['recon_s_t_w'] * self.loss_gen_recon_s_dw_t + \
                              hyperparameters['recon_s_p_w'] * self.loss_gen_recon_s_dw_p + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_dw + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_sc + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_dw + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_sc + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_dw
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_t_a1 = Variable(self.s_t_a)
        s_p_a1 = Variable(self.s_p_a)
        s_t_b1 = Variable(self.s_t_b)
        s_p_b1 = Variable(self.s_p_b)
        s_t_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_p_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_t_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_p_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_t_fake, s_a_p_fake = self.gen_sc.encode(x_a[i].unsqueeze(0))
            c_b, s_b_t_fake, s_b_p_fake = self.gen_dw.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.decode(
                c_a, 
                [s_a_t_fake, s_a_p_fake],
                self.gen_sc,
                [self.gen_sc, self.gen_sc]
            ))
            x_b_recon.append(self.decode(
                c_b, 
                [s_b_t_fake, s_b_p_fake],
                self.gen_dw,
                [self.gen_dw, self.gen_dw]
            ))
            x_ba1.append(self.decode(
                c_b,
                [s_t_a1[i].unsqueeze(0), s_p_a1[i].unsqueeze(0)],
                self.gen_dw,
                [self.gen_sc, self.gen_sc]
            ))
            x_ba2.append(self.decode(
                c_b, 
                [s_t_a2[i].unsqueeze(0), s_p_a2[i].unsqueeze(0)],
                self.gen_dw,
                [self.gen_sc, self.gen_sc]
            ))
            x_ab1.append(self.decode(
                c_a, 
                [s_t_b1[i].unsqueeze(0), s_p_b1[i].unsqueeze(0)],
                self.gen_sc,
                [self.gen_dw, self.gen_dw]
            ))
            x_ab2.append(self.decode(
                c_a, 
                [s_t_b2[i].unsqueeze(0), s_p_b2[i].unsqueeze(0)],
                self.gen_sc,
                [self.gen_dw, self.gen_dw]
            ))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_sc, x_dw, x_sw, hyperparameters):
        """
            Update discriminator.
        """
        self.dis_opt.zero_grad()
        s_t_sc = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_p_sc = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_t_dw = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_p_dw = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_t_sw = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_p_sw = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())

        # -------------------------encode------------------------------------
        c_sc, _, _ = self.gen_sc.encode(x_sc)
        c_dw, _, _ = self.gen_dw.encode(x_dw)
        c_sw, _, _ = self.gen_sw.encode(x_sw)

        # -------------------------decode (cross domain)-------------------------
        x_dw2sc = self.decode(
            c_dw, 
            [s_t_sc, s_p_sc],
            self.gen_dw, 
            [self.gen_sc, self.gen_sc]
        )
        x_sc2dw = self.decode(
            c_sc,
            [s_t_dw, s_p_dw],
            self.gen_sc,
            [self.gen_dw, self.gen_dw]
        )
        x_sc2sw = self.decode(
            c_sc,
            [s_t_dw, s_p_sc],
            self.gen_sc,
            [self.gen_dw, self.gen_sc]
        )
        x_sw2dw = self.decode(
            c_sw, 
            [s_t_sw, s_p_dw],
            self.gen_sw, 
            [self.gen_sw, self.gen_dw]
        )
        x_dw2sw_1 = self.decode(
            c_dw,
            [s_t_dw, s_p_sc],
            self.gen_dw,
            [self.gen_dw, self.gen_sc]
        )
        x_dw2sw_2 = self.decode(
            c_dw, 
            [s_t_dw, s_p_sw],
            self.gen_dw,
            [self.gen_dw, self.gen_sw]
        )
        x_sw2sc_1 = self.decode(
            c_sw,
            [s_t_sc, s_p_sw],
            self.gen_sw,
            [self.gen_sc, self.gen_sw]
        )
        x_sw2sc_2 = self.decode(
            c_sw,
            [s_t_sc, s_p_sc],
            self.gen_sw,
            [self.gen_sc, self.gen_sc]
        )

        # -------------------------D loss-------------------------
        self.loss_dis_a = self.dis_sc.calc_dis_loss(x_dw2sc.detach(), x_sc)
        self.loss_dis_b = self.dis_dw.calc_dis_loss(x_sc2dw.detach(), x_dw)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_sc.load_state_dict(state_dict['a'])
        self.gen_dw.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_sc.load_state_dict(state_dict['a'])
        self.dis_dw.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_sc.state_dict(), 'b': self.gen_dw.state_dict()}, gen_name)
        torch.save({'a': self.dis_sc.state_dict(), 'b': self.dis_dw.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def decode(self, content, style_codes:list, content_gen, style_gen:list):
        """ 
            Decode content and style codes to an image 
            style_codes && style_gen should in order : [texture, physic]
            Style decoder should be called before content decoder !
        """
        
        style_texture, style_physic = style_codes
        texture_decoder, physic_decoder = style_gen[0].texture_decoder, style_gen[1].physic_decoder
        # self.logger.info('---------------------decoding------------------------')
        adain_params_t = style_gen[0].mlp_texture(style_texture)
        adain_params_p = style_gen[1].mlp_physic(style_physic)
        style_gen[0].assign_decoder_AdaIn(adain_params_t, texture_decoder)
        style_gen[1].assign_decoder_AdaIn(adain_params_p, physic_decoder)
        # We split the MUNIT decoder
        # self.logger.info(f'content :{content.shape}')
        feature = texture_decoder(content)
        feature = physic_decoder(feature)
        images = content_gen.content_decoder(feature)
        return images