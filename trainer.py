"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis
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
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        # TODO: ------------------Core Change--------------
        display_size = int(hyperparameters['display_size'])
        self.s_t_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_t_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.logger = logging.getLogger(self.__class__.__name__)

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def  gen_update(self, x_a, x_b, hyperparameters):
        """
            # TODO: ------------------Core Change--------------
        """
        self.gen_opt.zero_grad()
        s_a_t = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a_p = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b_t = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b_p = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_t_prime, s_a_p_prime = self.gen_a.encode(x_a)
        c_b, s_b_t_prime, s_b_p_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.decode(
            c_a,
            [s_a_t_prime, s_a_p_prime],
            self.gen_a,
            [self.gen_a, self.gen_a]
        )
        x_b_recon = self.decode(
            c_b, 
            [s_b_t_prime, s_b_p_prime],
            self.gen_b,
            [self.gen_b, self.gen_b]
        )
        # decode (cross domain)
        x_ba = self.decode(
            c_b, 
            [s_a_t, s_a_p],
            self.gen_b,
            [self.gen_a, self.gen_a]
        )
        x_ab = self.decode(
            c_a, 
            [s_b_t, s_b_p],
            self.gen_a,
            [self.gen_b, self.gen_b]
        )
        # encode again
        c_b_recon, s_a_t_recon, s_a_p_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_t_recon, s_b_p_recon = self.gen_b.encode(x_ab)
        # decode again (if needed) 
        x_aba = self.decode(
            c_a_recon, 
            [s_a_t_prime, s_a_p_prime],
            self.gen_a,
            [self.gen_a, self.gen_a]
        ) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.decode(
            c_b_recon, 
            [s_b_t_prime, s_b_p_prime],
            self.gen_b,
            [self.gen_b, self.gen_b]
        ) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a_t = self.recon_criterion(s_a_t_recon, s_a_t)
        self.loss_gen_recon_s_a_p = self.recon_criterion(s_a_p_recon, s_a_p)
        self.loss_gen_recon_s_b_t = self.recon_criterion(s_b_t_recon, s_b_t)
        self.loss_gen_recon_s_b_p = self.recon_criterion(s_b_p_recon, s_b_p)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        # the last 4 items are not shown in paper.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_t_w'] * self.loss_gen_recon_s_a_t + \
                              hyperparameters['recon_s_p_w'] * self.loss_gen_recon_s_a_p + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_t_w'] * self.loss_gen_recon_s_b_t + \
                              hyperparameters['recon_s_p_w'] * self.loss_gen_recon_s_b_p + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
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
            c_a, s_a_t_fake, s_a_p_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_t_fake, s_b_p_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.decode(
                c_a, 
                [s_a_t_fake, s_a_p_fake],
                self.gen_a,
                [self.gen_a, self.gen_a]
            ))
            x_b_recon.append(self.decode(
                c_b, 
                [s_b_t_fake, s_b_p_fake],
                self.gen_b,
                [self.gen_b, self.gen_b]
            ))
            x_ba1.append(self.decode(
                c_b,
                [s_t_a1[i].unsqueeze(0), s_p_a1[i].unsqueeze(0)],
                self.gen_b,
                [self.gen_a, self.gen_a]
            ))
            x_ba2.append(self.decode(
                c_b, 
                [s_t_a2[i].unsqueeze(0), s_p_a2[i].unsqueeze(0)],
                self.gen_b,
                [self.gen_a, self.gen_a]
            ))
            x_ab1.append(self.decode(
                c_a, 
                [s_t_b1[i].unsqueeze(0), s_p_b1[i].unsqueeze(0)],
                self.gen_a,
                [self.gen_b, self.gen_b]
            ))
            x_ab2.append(self.decode(
                c_a, 
                [s_t_b2[i].unsqueeze(0), s_p_b2[i].unsqueeze(0)],
                self.gen_a,
                [self.gen_b, self.gen_b]
            ))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        """
            # TODO: ------------------Core Change--------------
        """
        self.dis_opt.zero_grad()
        s_t_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_p_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_t_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_p_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _, _ = self.gen_a.encode(x_a)
        c_b, _, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.decode(
            c_b, 
            [s_t_a, s_p_a],
            self.gen_b, 
            [self.gen_a, self.gen_a]
        )
        x_ab = self.decode(
            c_a,
            [s_t_b, s_p_b],
            self.gen_a,
            [self.gen_b, self.gen_b]
        )
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
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
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
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
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
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