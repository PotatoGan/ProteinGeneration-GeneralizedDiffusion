from comet_ml import Experiment
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

import numpy as np
from tqdm import tqdm
from einops import rearrange

import torchgeometry as tgm
import glob
import os
from PIL import Image

inputSize = 64

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cycle_cat(dl):
    while True:
        for data in dl:
            yield data[0]

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

import torch
import torchvision

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        device_of_kernel,
        channels = 3,
        timesteps = 1001,
        loss_type = 'l1',
        kernel_std = 0.1,
        kernel_size = 3,
        blur_routine = 'Incremental',
        train_routine = 'Final',
        sampling_routine='default',
        discrete = False,
        results_folder = 'GMM2',
        noiseLev = 0.15,
        indices_to_skip = []
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.kernel_std = kernel_std
        self.kernel_size = kernel_size
        self.blur_routine = blur_routine

        to_torch = partial(torch.tensor, dtype=torch.float32)
        #self.gaussian_kernels = nn.ModuleList(self.get_kernels())
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.discrete = discrete
        self.results_folder = Path(results_folder)
        self.noiseLev = noiseLev
        self.indices_to_skip = indices_to_skip

    def addNoise(self, img, i):
        torch.manual_seed(42 + i)
        #inputSize = img.size(0)  # Assuming inputSize is the size of the image
        noise = torch.randn(inputSize, inputSize).to(device='cuda')
        
        # 创建一个与噪声相同形状的掩码矩阵，初始化为1
        mask = torch.ones_like(noise)
        # 将需要跳过的位置设置为0
        mask[self.indices_to_skip, :] = 0
        
        # 将噪声与掩码相乘，以将需要跳过的位置置为0
        noise = noise * mask

        new_seed = torch.seed()  # 生成一个新的随机种子
        torch.manual_seed(new_seed)
        
        noise2 = torch.randn(inputSize, inputSize).to(device='cuda')

        img_with_noise = img + noise * self.noiseLev * i * 0.15 + noise2 * 0.035
        return img_with_noise

    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):
        
        #utils.save_image(img, str(self.results_folder / f'xt-0.png'), nrow=6)

        self.denoise_fn.eval()

        if t==None:
            t=self.num_timesteps

        if self.blur_routine == 'Individual_Incremental':
            #img = self.gaussian_kernels[t-1](img)
            x = self.addNoise(x, t - 1)
        
        else:
            for i in range(t):
                with torch.no_grad():
                    #img = self.gaussian_kernels[i](img)
                    img = self.addNoise(img, i)
                    #utils.save_image(img, str(self.results_folder / f'xt-{i}.png'), nrow=6)
        

        
        orig_mean = torch.mean(img, [2, 3], keepdim=True)
        #print(orig_mean.squeeze()[0])

        #print("gk")
        #print(self.gaussian_kernels)
        #self.print_kernels(self.gaussian_kernels)

        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while(t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)

            if self.train_routine == 'Final':
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == 'default':
                    if self.blur_routine == 'Individual_Incremental':
                        #x = self.gaussian_kernels[t - 2](x)
                        x = self.addNoise(x, t - 2)
                    else:
                        for i in range(t-1):
                            with torch.no_grad():
                                #x = self.gaussian_kernels[i](x)
                                x = self.addNoise(x, i)

                elif self.sampling_routine == 'x0_step_down':
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            #x_times = self.gaussian_kernels[i](x_times)
                            x_times = self.addNoise(x_times, i)
                            if self.discrete:
                                if i == (self.num_timesteps - 1):
                                    x_times = torch.mean(x_times, [2, 3], keepdim=True)
                                    x_times = x_times.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            #x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)
                            x_times_sub_1 = self.addNoise(x_times_sub_1, i)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1
            
            
        self.denoise_fn.train()
        return xt, direct_recons, img

    
    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):

        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t
        
        print(t)
        print(times)

        if self.blur_routine == 'Individual_Incremental':
            #img = self.gaussian_kernels[t - 1](img)
            img = self.addNoise(img, t - 1)
        else:
            for i in range(t):
                with torch.no_grad():
                    #img = self.gaussian_kernels[i](img)
                    img = self.addNoise(img, i)

        X_0s = []
        X_ts = []
        Y_0s = []
        
        """
        tempTimes = times
        tempImg = img
        while (tempTimes):
            step = torch.full((batch_size,), tempTimes - 1, dtype=torch.long).cuda()
            y = self.denoise_fn(tempImg, step)
            Y_0s.append(y)
            #print("y:", step)
            tempTimes = tempTimes - 1
        """
        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])
            noise = torch.randn_like(img) * 0.001
            img = img + noise

        # 3(2), 2(1), 1(0)
        while (times):
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            #print("x:", step)
            X_0s.append(x)
            X_ts.append(img)
            
            if self.train_routine == 'Final':
                if self.sampling_routine == 'default':
                    if self.blur_routine == 'Individual_Incremental':
                        if times-2 >= 0:
                            #x = self.gaussian_kernels[times - 2](img)
                            x = self.addNoise(img, times - 2)
                    else:
                        x_times_sub_1 = x
                        for i in range(times-1):
                            with torch.no_grad():
                                #x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)
                                x_times_sub_1 = self.addNoise(x_times_sub_1, i)

                        x = x_times_sub_1

                elif self.sampling_routine == 'x0_step_down':
                    if self.blur_routine == 'Individual_Incremental':
                        if times-2 >= 0:
                            #x = self.gaussian_kernels[times - 2](img)
                            x = self.addNoise(img, times - 2)
                    else:
                        x_times = x
                        for i in range(times):
                            with torch.no_grad():
                                #x_times = self.gaussian_kernels[i](x_times)
                                x_times = self.addNoise(x_times, i)
                                if self.discrete:
                                    if i == (self.num_timesteps - 1):
                                        x_times = torch.mean(x_times, [2, 3], keepdim=True)
                                        x_times = x_times.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])


                        x_times_sub_1 = x
                        for i in range(times - 1):
                            with torch.no_grad():
                                #x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)
                                x_times_sub_1 = self.addNoise(x_times_sub_1, i)

                        x = img - x_times + x_times_sub_1
                        #x = x - (noise/self.num_timesteps)



            img = x
            times = times - 1

        if self.discrete:
            img = img - noise
        X_0s.append(img)

        self.denoise_fn.train()
        return X_0s, X_ts, Y_0s
    

    def q_sample(self, x_start, t):
        # So at present we will for each batch blur it till the max in t.
        # And save it. And then use t to pull what I need. It is nothing but series of convolutions anyway.
        # Remember to do convs without torch.grad
        max_iters = torch.max(t)
        all_blurs = []
        x = x_start
        for i in range(max_iters+1):
            with torch.no_grad():
                #x = self.gaussian_kernels[i](x)
                x = self.addNoise(x, i)
                if self.discrete:
                    if i == (self.num_timesteps-1):
                        x = torch.mean(x, [2, 3], keepdim=True)
                        x = x.expand(x_start.shape[0], x_start.shape[1], x_start.shape[2], x_start.shape[3])
                all_blurs.append(x)

        all_blurs = torch.stack(all_blurs)

        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        for step in range(t.shape[0]):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])

        choose_blur = torch.stack(choose_blur)
        if self.discrete:
            choose_blur = (choose_blur + 1) * 0.5
            choose_blur = (choose_blur * 255)
            choose_blur = choose_blur.int().float() / 255
            choose_blur = choose_blur * 2 - 1
        #choose_blur = all_blurs
        return choose_blur


    def p_losses(self, x_start, t):
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        #print(t)
        return self.p_losses(x, t, *args, **kwargs)

class Dataset_Aug1(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
    
def protein_to_onehot(protein_sequence):
    # 定义氨基酸的种类，包括第21种
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    num_amino_acids = len(amino_acids)
    
    # 将蛋白质序列长度扩展到60，并将不足部分用第21种氨基酸补齐
    padded_sequence = protein_sequence.ljust(inputSize, amino_acids[-1])
    
    # 初始化一个全为0的onehot编码矩阵
    onehot = np.zeros((len(padded_sequence), num_amino_acids))
    
    # 将蛋白质序列中的氨基酸映射为对应的onehot编码
    for i, amino_acid in enumerate(padded_sequence):
        if amino_acid in amino_acids:
            onehot[i, amino_acids.index(amino_acid)] = 1
        else:
            # 如果碰到未知的氨基酸，则用全为0的向量表示
            pass
    
    # 将onehot编码改为正负1的形式
    onehot[onehot == 0] = -1
    new_array = -1 * np.ones((inputSize, inputSize))
    new_array[:inputSize, :21] = onehot
    reshaped_tensor = np.expand_dims(new_array, axis=0)
    return reshaped_tensor

class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取特征序列并转换为onehot编码
        feature_sequence = self.data[idx]
        onehot = protein_to_onehot(feature_sequence)
        # 转换为PyTorch的Tensor并返回
        return torch.tensor(onehot, dtype=torch.float32)

# trainer class
import os
import errno
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('.module', '')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = inputSize,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 100,
        results_folder = './results',
        load_path = None,
        dataset = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'mnist' or dataset == 'cifar10' or dataset == 'flower' or dataset == 'celebA' or dataset == 'AFHQ':
            print(dataset, "DA used")
            self.ds = Dataset_Aug1(folder, image_size)
            print(folder)
            self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=0,
                                drop_last=True))

        elif dataset == 'LSUN_train':
            print(dataset, "DA used")
            transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
            self.ds = torchvision.datasets.LSUN(root=folder, classes=['church_outdoor_train'], transform=transform)
            self.dl = cycle_cat(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=0,
                                drop_last=True))

        elif dataset == 'Protein':
            print(dataset)
            self.ds = ProteinDataset(folder)
            self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=0,
                                drop_last=True))
        else:
            print(dataset)
            self.ds = Dataset(folder, image_size)
            self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=0,
                                drop_last=True))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)
        
        # 去除 "module." 前缀
        new_state_dict = {key.replace("module.", ""): value for key, value in data['model'].items()}
        
        self.step = data['step']
        
        # 加载模型状态字典时，使用 strict=False 参数忽略不匹配的键名
        self.model.module.load_state_dict(new_state_dict, strict=False)
        
        # 加载 EMA 模型状态字典时，使用 strict=False 参数忽略不匹配的键名
        self.ema_model.load_state_dict(data['ema'], strict=False)



    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)



    def train(self):

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = torch.mean(self.model(data)) # change for DP
                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                    #self.print_kernel_parameters()
                    #print("data:")
                    #print(self.model(data))
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)
                #get_conv_weights(self.model)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.dl).cuda()
                xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, img=og_img) # change for DP

                og_img = (og_img + 1) * 0.5
                utils.save_image(og_img, str(self.results_folder / f'sample-og-{self.step}.png'), nrow=6)

                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-recon-{self.step}.png'), nrow = 6)

                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{self.step}.png'), nrow=6)

                xt = (xt + 1) * 0.5
                utils.save_image(xt, str(self.results_folder / f'sample-xt-{self.step}.png'),
                                 nrow=6)

                acc_loss = acc_loss/(self.save_and_sample_every+1)
                print(f'Mean of last {self.step}: {acc_loss}')
                acc_loss=0

                self.save()
                if self.step % (self.save_and_sample_every * 1) == 0:
                    self.save(self.step)

            self.step += 1

        print('training completed')


    def test_from_data(self, extra_path,d_times = None, s_times=None):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        X_0s, X_ts, Y_0s = self.ema_model.module.all_sample(batch_size=batches, img=og_img,t = d_times, times=s_times) # change for DP

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []
        frames_y = []

        for i in range(len(X_ts)):  #根据Issues的#9，len(X_ts) < len(X_0s)  Hence, this line should be changed to for i in range(len(X_ts)):
            #print(i)
            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))
            
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)
        
        for i in range(len(X_0s)):
            print(i)
            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))
        
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        
        return og_img, X_0s, X_ts
        
    def save_training_data(self):
        dataset = self.ds
        create_folder(f'{self.results_folder}/')

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = (img + 1) * 0.5
            utils.save_image(img, str(f'{self.results_folder}/' + f'{idx}.png'))
            if idx%1000 == 0:
                print(idx)
