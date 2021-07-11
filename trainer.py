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
    def __init__(self, opt):
        super(MultiStyle_Trainer, self).__init__()
        lr = opt['lr']
        # Initiate the networks
        # self.gen_sc = AdaINGen(opt['input_dim_sc'], opt['gen'])  # auto-encoder for domain sc
        # self.gen_dw = AdaINGen(opt['input_dim_dw'], opt['gen'])  # auto-encoder for domain dw
        # self.gen_sw = AdaINGen(opt['input_dim_sw'], opt['gen'])  # auto-encoder for domain sw
        self.gen = MultiStyle_Gen(opt['input_dim_sc'], opt['gen'])
        self.dis_sc = MsImageDis(opt['input_dim_sc'], opt['dis'])  # discriminator for domain sc
        self.dis_dw = MsImageDis(opt['input_dim_dw'], opt['dis'])  # discriminator for domain dw
        self.dis_sw = MsImageDis(opt['input_dim_sw'], opt['dis'])  # discriminator for domain sw
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = opt['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(opt['display_size'])
        self.s_t_sc = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_sc = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_t_dw = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_dw = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_t_sw = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_p_sw = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = opt['beta1']
        beta2 = opt['beta2']
        dis_params = list(self.dis_sc.parameters()) + list(self.dis_dw.parameters()) + list(self.dis_sw.parameters())
        # gen_params = list(self.gen_sc.parameters()) + list(self.gen_dw.parameters()) + list(self.gen_sw.parameters())
        gen_params = self.gen.parameters()
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=opt['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=opt['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, opt)
        self.gen_scheduler = get_scheduler(self.gen_opt, opt)

        # Network weight initialization
        self.apply(weights_init(opt['init']))
        self.dis_sc.apply(weights_init('gaussian'))
        self.dis_dw.apply(weights_init('gaussian'))
        self.dis_sw.apply(weights_init('gaussian'))

        self.logger = logging.getLogger(self.__class__.__name__)

        # Load VGG model if needed
        if 'vgg_w' in opt.keys() and opt['vgg_w'] > 0:
            self.vgg = load_vgg16(opt['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def image_recon_criterion(self, input, target):
        if not isinstance(input, list):
            input = [input]
        res = 0.
        for i in input:
            res += torch.mean(torch.abs(i - target))
        return res

    # def style_recon_criterion(self, input, target):
    #     if not isinstance(input, list):
    #         input = [input]
    #     res = 0.
    #     for i in input:
    #         res += torch.mean(torch.abs(i - target))
    #     return res

    def gen_update(self, x_sc, x_dw, x_sw, opt):
        """
            Update generator.
        """
        self.gen_opt.zero_grad()

        s_t_cloth_noise = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        # s_sc_p = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        # s_dw_t = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_p_dy_noise = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        # s_sw_t = Variable(torch.randn(x_sw.size(0), self.style_dim, 1, 1).cuda())
        # s_sw_p = Variable(torch.randn(x_sw.size(0), self.style_dim, 1, 1).cuda())

        # ------------------------encode-------------------------------------
        sc_info = {'p': 'static', 't': 'cloth'}
        dw_info = {'p': 'dynamic', 't': 'water'}
        sw_info = {'p': 'static', 't': 'water'}
        c_sc, s_sc_t_prime, s_sc_p_prime = self.gen.encode(x_sc, sc_info)
        c_dw, s_dw_t_prime, s_dw_p_prime = self.gen.encode(x_dw, dw_info)
        c_sw, s_sw_t_prime, s_sw_p_prime = self.gen.encode(x_sw, sw_info)

        if opt['style_pack'] == 'list':
            t_w_prime = [s_dw_t_prime, s_sw_t_prime]
            p_sa_prime = [s_sc_p_prime, s_sw_p_prime]
        elif opt['style_pack'] == 'random':
            pass
        else:
            raise NotImplementedError

        # ------------------------decode (dw)-----------------------
        # within domain
        dw_recon = []
        info = {'p': 'dynamic_code', 't': 'water_code'}
        for t in t_w_prime:
            cur_dw = self.gen.decode(
                c_dw,
                [t, s_dw_p_prime],
                info
            )
            dw_recon.append(cur_dw)

        # cross domain
        dw_cross = []
        info = {'p': 'dynamic_noise', 't': 'water_code'}
        for t in t_w_prime:
            cur_dw = self.gen.decode(
                c_sw,
                [t, s_p_dy_noise],
                info
            )
            dw_cross.append(cur_dw)

        # -------------------------decode (sw)-------------------------
        # within domain
        sw_recon = []
        info = {'p': 'static_code', 't': 'water_code'}
        for t in t_w_prime:
            for p in p_sa_prime:
                cur_sw = self.gen.decode(
                    c_sw, 
                    [t, p],
                    info
                )
                sw_recon.append(cur_sw)

        # cross domain
        sw_cross = []
        info = {'p': 'static_code', 't': 'water_code'}
        for t in t_w_prime:
            for p in p_sa_prime:
                cur_sw = self.gen.decode(
                    c_dw, 
                    [t, p],
                    info
                )
                sw_cross.append(cur_sw)  

        # -------------------------decode (sc)-------------------------
        # within domain
        sc_recon = []
        info = {'p': 'static_code', 't': 'cloth_code'}
        for p in p_sa_prime:
            cur_sc = self.gen.decode(
                c_sc,
                [s_sc_t_prime, p],
                info
            )
            sc_recon.append(cur_sc)

        # cross domain
        sc_cross = []
        info = {'p': 'static_code', 't': 'cloth_noise'}
        for p in p_sa_prime:
            cur_sc = self.gen.decode(
                c_dw, 
                [s_t_cloth_noise, p],
                info
            )
            sc_cross.append(cur_sc)

        for p in p_sa_prime:
            cur_sc = self.gen.decode(
                c_sw, 
                [s_t_cloth_noise, p],
                info
            )
            sc_cross.append(cur_sc)
        # -------------------------decode (dc)-------------------------
        # only cross domain
        dc_cross = []
        info = {'p': 'dynamic_noise', 't': 'cloth_noise'}
        cur_dc = self.gen.decode(
            c_sc, 
            [s_t_cloth_noise, s_p_dy_noise],
            info
        )
        dc_cross.append(cur_dc)

        cur_dc = self.gen.decode(
            c_dw, 
            [s_t_cloth_noise, s_p_dy_noise],
            info
        )
        dc_cross.append(cur_dc)
        
        cur_dc = self.gen.decode(
            c_sw, 
            [s_t_cloth_noise, s_p_dy_noise],
            info
        )
        dc_cross.append(cur_dc)

        # --------------------------style recon && loss------------------------------------
        # We use avg of these non-random style to supervise style recon.
        # At the same time, non-random style should supervise L1 loss.
        tw_avg = (s_dw_t_prime + s_sw_t_prime) / 2
        sa_avg = (s_sc_p_prime + s_sw_p_prime) / 2

        c_cloth_recon, c_sw_recon, c_dw_recon = [], [], []
        t_cloth_recon, t_water_recon = [], []
        p_dynamic_recon, p_static_recon = [], []

        # First, within domain content && style (mostly non-random )
        for sc in sc_recon:
            info = {'p': 'static', 't': 'water'}
            c, st, sp = self.gen.encode(sc, info) 
            c_cloth_recon.append(c)
            t_cloth_recon.append(st)
            p_static_recon.append(sp)

        for dw in dw_recon:
            info = {'p': 'dynamic', 't': 'water'}
            c, st, sp = self.gen.encode(dw, info)  # TODO: maybe not pack like this
            c_dw_recon.append(c)
            t_water_recon.append(st)
            p_dynamic_recon.append(sp)

        for sw in sw_recon:
            info = {'p': 'static', 't': 'water'}
            c, st, sp = self.gen.encode(sw, info)
            c_sw_recon.append(c)
            t_water_recon.append(st)
            p_static_recon.append(sp)

        self.loss_gen_recon_c_sc = self.recon_criterion(c_cloth_recon, c_sc)
        self.loss_gen_recon_c_dw = self.recon_criterion(c_dw_recon, c_dw)
        self.loss_gen_recon_c_sw = self.recon_criterion(c_sw_recon, c_sw)

        self.loss_gen_recon_t_cloth = self.recon_criterion(t_cloth_recon, s_sc_t_prime)
        self.loss_gen_recon_t_water = self.recon_criterion(t_cloth_recon, tw_avg)

        self.loss_gen_recon_p_dynamic = self.recon_criterion(p_dynamic_recon, self.gen.mlp_physic_dynamic(s_p_dy_noise))
        self.loss_gen_recon_p_static = self.recon_criterion(p_static_recon, sa_avg)

        # Second, cross domain style (mostly random)
        for dw in dw_cross:
            info = {'p': 'dynamic', 't': 'water'}
            c, st, sp = self.gen.encode(dw, info)
            self.loss_gen_recon_c_sw += self.recon_criterion(c, c_sw)
            self.loss_gen_recon_t_water += self.recon_criterion(st, tw_avg)
            self.loss_gen_recon_p_dynamic = self.recon_criterion(sp, self.gen.mlp_physic_dynamic(s_p_dy_noise))
        
        for sw in sw_cross:
            info = {'p': 'static', 't': 'water'}
            c, st, sp = self.gen.encode(sw, info)
            self.loss_gen_recon_c_dw += self.recon_criterion(c, c_dw)
            self.loss_gen_recon_t_water += self.recon_criterion(st, tw_avg)
            self.loss_gen_recon_p_static += self.recon_criterion(sp, sa_avg)

        for sc in sc_cross[:len(sc_cross) // 2]:
            info = {'p': 'static', 't': 'cloth'}
            c, st, sp = self.gen.encode(sc, info)  # dw -> sc
            self.loss_gen_recon_c_dw += self.recon_criterion(c, c_dw)
            self.loss_gen_recon_t_cloth += self.recon_criterion(st, self.gen.mlp_texture_cloth(s_t_cloth_noise))
            self.loss_gen_recon_p_static += self.recon_criterion(st, sa_avg)

        for sc in sc_cross[len(sc_cross) // 2:]:
            info = {'p': 'static', 't': 'cloth'}
            c, st, sp = self.gen.encode(sc, info)  # sw -> sc
            self.loss_gen_recon_c_sw += self.recon_criterion(c, c_sw)
            self.loss_gen_recon_t_cloth = self.recon_criterion(st, self.gen.mlp_texture_cloth(s_t_cloth_noise))
            self.loss_gen_recon_p_static += self.recon_criterion(st, sa_avg)

#------------------TODO: go to master branch to verify MUNIT's wrong L1 supervision on style noise.-----------------------
        # recon dc's content && style.  
        # Order : [sc, dw, sw]
        info = {'p': 'dynamic', 't': 'cloth'}
        # c_sc in dc_cross
        c, st, sp = self.gen.encode(dc_cross[0], info) 
        self.loss_gen_recon_c_sc += self.recon_criterion(c, c_sc)
        self.loss_gen_recon_t_cloth += self.recon_criterion(t_cloth_recon, s_sc_t_prime)
        self.loss_gen_recon_p_dynamic += self.recon_criterion(sp, self.gen.mlp_physic_dynamic(s_p_dy_noise))

        # c_dw in dc_cross
        c, st, sp = self.gen.encode(dc_cross[1], info) 
        self.loss_gen_recon_c_dw += self.recon_criterion(c, c_dw)
        self.loss_gen_recon_t_cloth += self.recon_criterion(t_cloth_recon, s_sc_t_prime)
        self.loss_gen_recon_p_dynamic += self.recon_criterion(sp, self.gen.mlp_physic_dynamic(s_p_dy_noise))

        # c_sw in dc_cross
        c, st, sp = self.gen.encode(dc_cross[2], info) 
        self.loss_gen_recon_c_sw += self.recon_criterion(c, c_sw)
        self.loss_gen_recon_t_cloth += self.recon_criterion(t_cloth_recon, s_sc_t_prime)
        self.loss_gen_recon_p_dynamic += self.recon_criterion(sp, self.gen.mlp_physic_dynamic(s_p_dy_noise))

#------------------TODO: go to master branch to verify MUNIT's wrong L1 supervision on style noise.-----------------------


        # ------------------------loss----------------------------
        # image recon loss
        self.loss_gen_recon_x_sc = self.image_recon_criterion(sc_recon, x_sc)
        self.loss_gen_recon_x_dw = self.image_recon_criterion(dw_recon, x_dw)
        self.loss_gen_recon_x_sw = self.image_recon_criterion(sw_recon, x_sw)

        # non-random style should be same
        self.t_water_loss = self.recon_criterion(s_dw_t_prime, s_sw_t_prime)
        self.p_sa_loss = self.recon_criterion(s_sc_p_prime, s_sw_p_prime)

        # ---------------------------GAN loss------------------------------------
        self.loss_gen_adv_sc = sum([self.dis_sc.calc_gen_loss(i) for i in sc_cross])
        self.loss_gen_adv_dw = sum([self.dis_dw.calc_gen_loss(i) for i in dw_cross])
        self.loss_gen_adv_sw = sum([self.dis_sw.calc_gen_loss(i) for i in sw_cross])

        # --------------------------domain-invariant perceptual loss------------------------
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if opt['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if opt['vgg_w'] > 0 else 0

        # --------------------------total loss--------------------------------------------------
        # the last 4 items are not shown in paper.
        self.loss_gen_total = \
            opt['gan_w'] * (self.loss_gen_adv_sc + self.loss_gen_adv_dw + self.loss_gen_adv_sw) + \
            opt['recon_x_w'] * (self.loss_gen_recon_x_sc + self.loss_gen_recon_x_dw + self.loss_gen_recon_x_sw) + \
            opt['same_style_w'] * (self.t_water_loss + self.p_sa_loss) + \
            opt['recon_c_w'] * (self.loss_gen_recon_c_sc + self.loss_gen_recon_c_dw + self.loss_gen_recon_c_sw) + \
            opt['recon_s_w'] * (self.loss_gen_recon_t_cloth + self.loss_gen_recon_t_water) + \
            opt['recon_s_w'] * (self.loss_gen_recon_p_dynamic + self.loss_gen_recon_p_static)
                              
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample_multi(self, x_sc, x_dw, x_sw):
        self.eval()

        s_sc_t = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_sc_p = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_dw_t = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_dw_p = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_sw_t = Variable(torch.randn(x_sw.size(0), self.style_dim, 1, 1).cuda())
        s_sw_p = Variable(torch.randn(x_sw.size(0), self.style_dim, 1, 1).cuda())

        # this recon is slightly different from gen_update
        # only recon itself
        # CROSS only use one item in the list
        sc_recon, dw_recon, sw_recon = [], [], []
        dc_cross, sc_cross, dw_cross, sw_cross = [], [], [], []
        for i in range(x_sc.size(0)):
            c_sc, s_sc_t_fake, s_sc_p_fake = self.gen.encode(
                x_sc[i].unsqueeze(0), 
                {'t': 'cloth', 'p': 'static'})
            c_dw, s_dw_t_fake, s_dw_p_fake = self.gen.encode(
                x_dw[i].unsqueeze(0),
                {'t': 'water', 'p': 'dynamic'})
            c_sw, s_sw_t_fake, s_sw_p_fake = self.gen.encode(
                x_sw[i].unsqueeze(0),
                {'t': 'water', 'p': 'static'})

            sc_recon.append(self.gen.decode(
                c_sc, 
                [s_sc_t_fake, s_sc_p_fake],
                {'t': 'cloth', 'p': 'static'}
            ))
            dw_recon.append(self.gen.decode(
                c_dw, 
                [s_dw_t_fake, s_dw_p_fake],
                {'t': 'water', 'p': 'dynamic'}
            ))
            sw_recon.append(self.gen.decode(
                c_sw, 
                [s_sw_t_fake, s_sw_p_fake],
                {'t': 'water', 'p': 'static'}
            ))

            # self.logger.info(f's_sc_t[i] : {s_sc_t[i].shape}')
            # self.logger.info(f's_sc_p[i] : {s_sc_p[i].shape}')
            sc_cross.append(self.gen.decode(
                c_dw,
                [s_sc_t[i].unsqueeze(0), s_sc_p_fake.unsqueeze(0)],
                {'t': 'cloth', 'p': 'static'}
            ))
            dw_cross.append(self.gen.decode(
                c_sw,
                [s_dw_t_fake.unsqueeze(0), s_dw_p[i].unsqueeze(0)],
                {'t': 'water', 'p': 'dynamic'}
            ))
            sw_cross.append(self.gen.decode(
                c_dw,
                [s_sw_t_fake.unsqueeze(0), s_sw_p_fake.unsqueeze(0)],
                {'t': 'water', 'p': 'static'}
            ))
            #TODO: import thing is here!
            dc_cross.append(self.gen.decode(
                c_dw,
                [s_sc_t_fake.unsqueeze(0), s_dw_p.unsqueeze(0)],
                {'t': 'water', 'p': 'static'}
            ))

        sc_recon, dw_recon, sw_recon = torch.cat(sc_recon), torch.cat(dw_recon), torch.cat(sw_recon)
        sc_cross, dw_cross, sw_cross = torch.cat(sc_cross), torch.cat(dw_cross), torch.cat(sw_cross)
        self.train()
        return  x_sc, sc_recon, sc_cross, \
                x_dw, dw_recon, dw_cross, \
                x_sw, sw_recon, sw_cross

    def dis_update(self, x_sc, x_dw, x_sw, opt):
        """
            Update discriminator.
        """
        self.dis_opt.zero_grad()
        s_sc_t = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_sc_p = Variable(torch.randn(x_sc.size(0), self.style_dim, 1, 1).cuda())
        s_dw_t = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_dw_p = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_sw_t = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())
        s_sw_p = Variable(torch.randn(x_dw.size(0), self.style_dim, 1, 1).cuda())

        # -------------------------encode------------------------------------
        sc_info = {'p': 'static', 't': 'cloth'}
        dw_info = {'p': 'dynamic', 't': 'water'}
        sw_info = {'p': 'static', 't': 'water'}
        c_sc, s_sc_t_prime, s_sc_p_prime = self.gen.encode(x_sc, sc_info)
        c_dw, s_dw_t_prime, s_dw_p_prime = self.gen.encode(x_dw, dw_info)
        c_sw, s_sw_t_prime, s_sw_p_prime = self.gen.encode(x_sw, sw_info)

        if opt['style_pack'] == 'list':
            t_w_prime = [s_dw_t_prime, s_sw_t_prime]
            p_sa_prime = [s_sc_p_prime, s_sw_p_prime]
        elif opt['style_pack'] == 'random':
            pass
        else:
            raise NotImplementedError

        # We only need cross domain transfer here
        # -------------------------decode (dw)-------------------------
        dw_cross = []
        info = {'p': 'dynamic', 't': 'water'}
        for t in t_w_prime:
            cur_dw = self.gen.decode(
                c_sw,
                [t, s_dw_p],
                info
            )
            dw_cross.append(cur_dw)

        # -------------------------decode (sw)-------------------------
        sw_cross = []
        info = {'p': 'static', 't': 'water'}
        for t in t_w_prime:
            for p in p_sa_prime:
                cur_sw = self.gen.decode(
                    c_dw, 
                    [t, p],
                    info
                )
                sw_cross.append(cur_sw)  

        # -------------------------decode (sc)-------------------------
        sc_cross = []
        info = {'p': 'static', 't': 'cloth'}
        for p in p_sa_prime:
            cur_sw = self.gen.decode(
                c_dw, 
                [s_sc_t, p],
                info
            )
            sc_cross.append(cur_sw)

        # -------------------------D loss-------------------------
        self.loss_dis_dw = sum([self.dis_dw.calc_dis_loss(i, x_dw) for i in dw_cross])
        self.loss_dis_sw = sum([self.dis_sw.calc_dis_loss(i, x_sw) for i in sw_cross])
        self.loss_dis_sc = sum([self.dis_sc.calc_dis_loss(i, x_sc) for i in sc_cross])
        self.loss_dis_total = self.loss_dis_dw + self.loss_dis_sw + self.loss_dis_sc
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, opt):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
     #   self.gen_sc.load_state_dict(state_dict['a'])
      #  self.gen_dw.load_state_dict(state_dict['b'])
        self.gen.load_state_dict(state_dict['multistyle-gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_sc.load_state_dict(state_dict['sc'])
        self.dis_dw.load_state_dict(state_dict['dw'])
        self.dis_sw.load_state_dict(state_dict['sw'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, opt, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, opt, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
      #  torch.save({'a': self.gen_sc.state_dict(), 'b': self.gen_dw.state_dict()}, gen_name)
        torch.save({'multistyle-gen': self.gen.state_dict()}, gen_name)
        torch.save(
            {'sc': self.dis_sc.state_dict(), 
            'dw': self.dis_dw.state_dict(),
            'sw': self.dis_sw.state_dict()}, 
            dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
