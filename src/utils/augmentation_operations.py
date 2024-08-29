"""Collection of utility functions"""
from scipy.fftpack import fft
# from utils.constants import *

import random
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import sys
from   torch.autograd import Variable
from   scipy.interpolate import CubicSpline
from   scipy.interpolate import interp1d
from   pyts.approximation import DiscreteFourierTransform
from   utils.utils import *

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# ----------------------------------Data Augmentation----------------------------------------
def random_curve_generator(ts, magnitude=.1, order=16, noise=None):
    seq_len = ts.shape[-1]
    x  = np.linspace(-seq_len, (2*ts.shape[0]-1) * seq_len - 1, 3 * ts.shape[0] * (seq_len//order), dtype=int)
    x2 = np.random.normal(loc=1.0, scale=magnitude, size=len(x))
    f  = CubicSpline(x, x2, axis=-1)
    return f(np.arange(seq_len))

def random_cum_curve_generator(ts, magnitude=.1, order=16, noise=None):
    x = random_curve_generator(ts, magnitude=magnitude, order=order, noise=noise).cumsum()
    x -= x[0]
    x /= x[-1]
    x = np.clip(x, 0, 1)
    return x * (ts.shape[-1] - 1)

def random_cum_noise_generator(ts, magnitude=.1, noise=None):
    seq_len = ts.shape[-1]
    x = (np.ones(seq_len) + np.random.normal(loc=0, scale=magnitude, size=seq_len)).cumsum()
    x -= x[0]
    x /= x[-1]
    x = np.clip(x, 0, 1)
    return x * (ts.shape[-1] - 1)

# 1
def _magnoise(x, pos_num=1, magnitude_min=0, magnitude_max=0.3, M=1, split=10, add=True):
    magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude <= 0: return x
    noise = torch.normal(mean=0, std=torch.ones(x.shape)*magnitude, out=None)
    if add:
        output = x + noise
        return output
    else:
        output = x * (1 + noise)
        return output

# 2
def _dimmagscale(x, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, magnitude=None):
    '''This tfm applies magscale to each dimension independently'''
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device)*(magnitude_max-magnitude_min)
    
    if magnitude <= 0: return x
    
    x     = x.reshape(x.shape[0],-1,3,x.shape[-1])
    scale = (1 + torch.rand((x.shape[0], x.shape[1], 1, 1)) * magnitude)
    if torch.rand(1) < .5: scale = 1 / scale
    output = x * scale
    output = output.reshape(x.shape[0], -1, x.shape[-1])
    return output

# 3
def _magwarp(x, pos_num=1, magnitude_min=0, magnitude_max=.4, M=5, split=10, order=16, magnitude=None):
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device)
    if magnitude <= 0: return x
    pos_chnnl = x.shape[1]//pos_num
    for i in range(pos_num):
        y_mult_mid = random_curve_generator(x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], magnitude=magnitude, order=order)
        output_mid = x[:,i*pos_chnnl:(i+1)*pos_chnnl,:] * x[:,i*pos_chnnl:(i+1)*pos_chnnl,:].new(y_mult_mid)
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

# 4
def _timewarp(x, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, order=16, magnitude=None):
    '''This is a slow batch tfm'''
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device)*(magnitude_max-magnitude_min)
    if magnitude <= 0: return x
    seq_len = x.shape[-1]
    pos_chnnl = x.shape[1]//pos_num
    for i in range(pos_num):
        f = CubicSpline(np.arange(seq_len), x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], axis=-1)
        new_x = random_cum_curve_generator(x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], magnitude=magnitude, order=order)
        output_mid = x[:,i*pos_chnnl:(i+1)*pos_chnnl,:].new(f(new_x))
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

# 5
def _timenoise(x, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, magnitude=None):
    '''This is a slow batch tfm'''
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device)*(magnitude_max-magnitude_min)
    if magnitude <= 0: return x
    pos_chnnl = x.shape[1]//pos_num
    for i in range(pos_num):
        f          = CubicSpline(np.arange(x.shape[-1]), x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], axis=-1)
        new_x      = random_cum_noise_generator(x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], magnitude=magnitude)
        output_mid = x.new(f(new_x))
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

def _zoomin(x, pos_num=1, magnitude=.8):
    '''This is a slow batch tfm'''
    if magnitude == 0: return x
    seq_len = x.shape[-1]
    pos_chnnl = x.shape[1]//pos_num
    for i in range(pos_num):
        lambd = np.random.beta(magnitude, magnitude)
        lambd = max(lambd, 1 - lambd)
        win_len = int(seq_len * lambd)
        if win_len == seq_len: start = 0
        else: start = np.random.randint(0, seq_len - win_len)
        x2 = x[:,i*pos_chnnl:(i+1)*pos_chnnl,start : start + win_len]
        f = CubicSpline(np.arange(x2.shape[-1]), x2, axis=-1)
        output_mid = x[:,i*pos_chnnl:(i+1)*pos_chnnl,:].new(f(np.linspace(0, win_len - 1, num=seq_len)))
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

def _zoomout(x, pos_num=1, magnitude=.8):
    '''This is a slow batch tfm'''
    if magnitude == 0: return x
    seq_len = x.shape[-1]
    pos_chnnl = x.shape[1]//pos_num
    for i in range(pos_num):
        lambd = np.random.beta(magnitude, magnitude)
        lambd = max(lambd, 1 - lambd)
        f = CubicSpline(np.arange(seq_len), x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], axis=-1)
        new_x = torch.zeros_like(x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], dtype=x.dtype, device=x.device)
        win_len = int(seq_len * lambd)
        new_x[..., -win_len:] = x[:,i*pos_chnnl:(i+1)*pos_chnnl,:].new(f(np.linspace(0, seq_len - 1, num=win_len)))
        output_mid = new_x
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

# 6
def _randomzoom(x, pos_num=1, magnitude_min=0, magnitude_max=0.8, M=10, split=10, magnitude=None):
    if magnitude <= 0.5:
        return _zoomin(x)
    else: 
        return _zoomout(x)

# 7
def _timestepzero(x, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10):
    magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude >= 1 or magnitude <= 0: return x
    seq_len = x.shape[-1]
    sen_num = int(x.shape[-2] / 3)
    input_channel = x.shape[-2]
    mask = np.random.choice([0, 1], (sen_num, seq_len), p=[magnitude, 1-magnitude])
    mask = mask.reshape(sen_num, 1, seq_len)
    mask = mask.repeat(3, axis=1)
    mask = mask.reshape(1, input_channel, seq_len)
    output = (x*mask).float()
    return output

# 8
def _cutout(x, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, magnitude=None):
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device).numpy()*(magnitude_max-magnitude_min)
    if magnitude >= 1 or magnitude <= 0: return x
    seq_len = x.shape[-1]
    sen_num = int(x.shape[-2] / 3)
    for i in range(sen_num):
        new_x = x[:,i*3:(i+1)*3,:].clone()
        win_len = int(magnitude * seq_len)
        start = np.random.randint(-win_len + 1, seq_len)
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        new_x[..., start:end] = 0
        output_mid = new_x
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

# 9
def _randomcrop(x, pos_num=1, magnitude_min=0, magnitude_max=.8, M=5, split=10, magnitude=None):
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device).numpy()*(magnitude_max-magnitude_min)
    if magnitude >= 1 or magnitude <= 0: return x
    seq_len = x.shape[-1]
    sen_num = int(x.shape[-2] / 3)
    for i in range(sen_num):
        lambd = np.random.beta(magnitude, magnitude)
        lambd = max(lambd, 1 - lambd)
        win_len = int(seq_len * lambd)
        if win_len == seq_len: 
            start = 0
        else: 
            start = np.random.randint(0, seq_len - win_len)
        end = start + win_len
        new_x = torch.zeros_like(x[:,i*3:(i+1)*3,:], dtype=x.dtype, device=x.device)
        new_x[..., start : end] = x[:,i*3:(i+1)*3, start : end]
        output_mid = new_x
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output # if y is None else [output, y]

# 10
def _timeshift(x, pos_num=1, magnitude_min=0, magnitude_max=.8, M=5, split=10, magnitude=None):
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(x.device).numpy()*(magnitude_max-magnitude_min)
    if magnitude >= 1 or magnitude <= 0: return x
    seq_len = x.shape[-1]
    pos_chnnl = x.shape[1]//pos_num
    for i in range(pos_num):
        shift_dis = int(np.random.randint(0, seq_len//2)*magnitude)
        new_x = torch.zeros_like(x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], dtype=x.dtype, device=x.device)
        if np.random.rand() <= 0.5:
            new_x[..., shift_dis: ] = x[:, i*pos_chnnl:(i+1)*pos_chnnl, 0:(seq_len-shift_dis)]
        else:
            new_x[..., 0:(seq_len-shift_dis)] = x[:, i*pos_chnnl:(i+1)*pos_chnnl, shift_dis:(seq_len+1)]
        output_mid = new_x
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return output

def generate_random_rotation_matrix(N):
    axis_xy = np.ones((N, 1))
    axis = np.zeros((N, 3))

    col_idx = np.random.choice([0, 1], size=(N,))
    axis[np.arange(N), col_idx] = axis_xy.squeeze()
    
    # Generate random angles between -pi and pi
    angle = np.random.uniform(-np.pi, np.pi, size=N)

    # Compute the rotation matrices using the axis-angle formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis.T
    R = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                  [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                  [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])
    return R.transpose((2, 0, 1))

# 11
def _randrotate(x, pos_num=1, magnitude_min=0, magnitude_max=0.1, M=5, split=10):
    magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude <= 0: return x
    input_channel = x.shape[1]
    seq_len = x.shape[-1]
    x = x.reshape(pos_num, -1, 3, seq_len)
    mat = generate_random_rotation_matrix(pos_num).reshape(pos_num,1,3,3)
    x = np.matmul(mat, x)
    x = x.reshape(1, input_channel, seq_len).float()
    return x

# 12
def _scaling(x, pos_num=1, magnitude_min=0, magnitude_max=1, M=1, split=10):
    magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude <= 0: return x
    
    x     = x.reshape(x.shape[0],-1,3,x.shape[-1])
    scale = torch.normal(mean=1, std=torch.ones((x.shape[0], x.shape[1], 1, x.shape[-1]))*magnitude, out=None)
    output = x * scale.to(x.device)
    output = output.reshape(x.shape[0], -1, x.shape[-1])
    
    return output

# 13
def _permutation(x, pos_num=1, max_segments=20, seg_mode="random"):
    timestamps   = np.arange(x.shape[-1])
    pos_chnnl    = x.shape[1]//pos_num
    max_segments = x.shape[-1]//max_segments
    num_pos      = np.random.randint(1, (max_segments+1), size=(pos_num))
    output       = np.zeros_like(x)
    # for i, pat in enumerate(x):
    for i in range(pos_num):
        if num_pos[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[-1] - 2, num_pos[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(timestamps, split_points)
            else:
                splits = np.array_split(timestamps, num_pos[i])
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            output[:,i*pos_chnnl:(i+1)*pos_chnnl,:] = x[:,i*pos_chnnl:(i+1)*pos_chnnl,warp]
        else:
            output[:,i*pos_chnnl:(i+1)*pos_chnnl,:] = x[:,i*pos_chnnl:(i+1)*pos_chnnl,:]
    return torch.from_numpy(output)

# 14
def _resample(x, pos_num=1):
    orig_steps   = np.arange(x.shape[-1])
    interp_steps = np.arange(0, orig_steps[-1]+0.001, 1/3)
    pos_chnnl    = x.shape[1]//pos_num
    for i in range(pos_num):
        Interp = interp1d(orig_steps, x[:,i*pos_chnnl:(i+1)*pos_chnnl,:], axis=-1)
        InterpVal = Interp(interp_steps)
        start = random.choice(orig_steps)
        resample_index = np.arange(start, 3 * x.shape[-1], 2)[:x.shape[-1]]
        output_mid     = InterpVal[:, :, resample_index]
        if i == 0:
            output = output_mid
        else:
            output = np.concatenate((output, output_mid), axis=1)
    return torch.from_numpy(output).float()

def distance(i, j, imageSize, r):
    dis_x = np.sqrt((i - imageSize[0] / 2) ** 2)
    dis_y =  np.sqrt((j - imageSize[1] / 2) ** 2)
    if dis_x < r[0] and dis_y < r[1]:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = torch.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=(rows, cols), r=r)
    return mask

# 15
def _generate_high(sample, pos_num=1, r=(32,2)):
    # r: int, radius of the mask
    images = torch.unsqueeze(sample, 1)
    images = images.permute(0,1,3,2)
    mask = mask_radial(torch.zeros([images.shape[2], images.shape[3]]), r)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1))) # shift: low f in the center
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * (1.-mask)
    fft = torch.real(fd)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = torch.squeeze(fd.reshape([bs, c, h, w]),dim=1)
    # return fft, fd
    return fd.permute(0,2,1)

# 16
def _generate_low(sample, pos_num=1, r=(32,2)):
    # r: int, radius of the mask
    images = torch.unsqueeze(sample, 1)
    images = images.permute(0,1,3,2)
    mask = mask_radial(torch.zeros([images.shape[2], images.shape[3]]), r)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1))) # shift: low f in the center
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * mask
    fft = torch.real(fd)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = torch.squeeze(fd.reshape([bs, c, h, w]),dim=1)
    # return fft, fd
    return fd.permute(0,2,1)

# 17
def _ifft_amp_noise(sample, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, magnitude=None):
    # magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(sample.device) * (magnitude_max-magnitude_min)
    if magnitude >= 1 or magnitude <= 0: return x
    
    sample = sample.permute(0,2,1)
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    
    mask  = np.random.choice([0, 1], fd.shape, p=[magnitude, 1-magnitude])  # magnitude代表选0的几率
    noise = torch.normal(mean=1, std=torch.ones(mask.shape)*magnitude, out=None)*mask
    noise = noise+(1-mask)
    fd    = fd * noise
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))).reshape([bs, c, h, w]), axis=1)

    return ifft.permute(0,2,1).float()

# 18
def _ifft_phase_shift(sample, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, magnitude=None):
    
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(sample.device).numpy() * (magnitude_max-magnitude_min)
    
    sample = sample.permute(0,2,1)
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp = fd.abs()
    phase = fd.angle()

    # phase shift
    angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, 
                       size=(sample.shape[0], sample.shape[1])), axis=2)*magnitude, sample.shape[2], axis=2)
    phase = phase + angles

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]), axis=1)

    return ifft.permute(0,2,1).float()

# 18
def _ifft_amp_shift(sample, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, magnitude=None):
    
    if magnitude==None:
        magnitude = M*(magnitude_max-magnitude_min)/split
    else:
        magnitude = magnitude.data.to(sample.device)*(magnitude_max-magnitude_min)
    
    sample = sample.permute(0,2,1)
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x  = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp   = fd.abs()
    phase = fd.angle()

    # amp shift
    amp    = amp + np.random.normal(loc=0., scale=magnitude, size=sample.shape)

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]), axis=1)

    return ifft.permute(0,2,1).float()

# 19
def _ifft_amp_phase_shift_win(sample, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10, win_size_rate=0.2):
    
    magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude >= 1 or magnitude <= 0: return x
    
    sample = sample.permute(0,2,1)
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp = fd.abs()
    phase = fd.angle()

    # select a segment to conduct perturbations
    start = np.random.randint(0, int(win_size_rate * sample.shape[1]))
    end = start + int(win_size_rate * sample.shape[1])
    # phase shift
    angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, 
                       size=(sample.shape[0], sample.shape[1])), axis=2)*magnitude, sample.shape[2], axis=2)
    phase[:, start:end, :] = phase[:, start:end, :] + angles[:, start:end, :]
    # amp shift
    amp[:, start:end, :] = amp[:, start:end, :] + np.random.normal(loc=0., scale=magnitude, size=sample.shape)[:, start:end, :]

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]), axis=1)

    return ifft.permute(0,2,1).float()

# 20
def _ifft_amp_phase_shift_fully(sample, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10):
    
    magnitude   = M*(magnitude_max-magnitude_min)/split
    if magnitude >= 1 or magnitude <= 0: return x
    
    sample      = sample.permute(0,2,1)
    images      = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x           = images.reshape([bs * c, h, w])
    fd          = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp    = fd.abs()
    phase  = fd.angle()

    # phase shift
    angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, 
                       size=(sample.shape[0], sample.shape[1])), axis=2)*magnitude, sample.shape[2], axis=2)
    phase  = phase + angles

    # amp shift
    amp    = amp + np.random.normal(loc=0., scale=magnitude, size=sample.shape)

    cmp    = amp * torch.exp(1j * phase)
    ifft   = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]), axis=1)

    return ifft.permute(0,2,1).float()

# 21
def _window_slice_overall(x, pos_num=1, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio*x.shape[2]).astype(int)
    if target_len >= x.shape[2]:
        return x
    starts = np.random.randint(low=0, high=x.shape[2]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    output = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[1]):
            output[i,dim,:] = np.interp(np.linspace(0, target_len, num=x.shape[2]), np.arange(target_len), pat[dim,starts[i]:ends[i]]).T
    return output

# 22
def _window_slice_pos(x, pos_num=1, reduce_ratio=0.9):
    
    pos_chnnl     = x.shape[1]//pos_num
    output = np.zeros_like(x)
    
    for i, pat in enumerate(x):
        for pos in range(pos_num):
            
            target_len = np.ceil(reduce_ratio*x.shape[2]).astype(int)
            if target_len >= x.shape[2]:
                return x
            starts = np.random.randint(low=0, high=x.shape[2]-target_len, size=(x.shape[0])).astype(int)
            ends = (target_len + starts).astype(int)
            
            Interp_win = interp1d(np.arange(target_len), pat[pos*pos_chnnl:(pos+1)*pos_chnnl,starts[i]:ends[i]], axis=-1)
            window_interp_steps = np.arange(0, target_len+0.001, target_len/x.shape[-1])[:x.shape[-1]]
            if window_interp_steps[-1] > (target_len-1):
                window_interp_steps[-1] = (target_len-1)
            InterpVal_win = Interp_win(window_interp_steps)
            output[i,pos*pos_chnnl:(pos+1)*pos_chnnl,:] = InterpVal_win
            
    return output

# 23
def _window_warp_overall(x, pos_num=1, window_ratio=0.1, scales=[0.5, 2.]):
    warp_scales   = np.random.choice(scales, x.shape[0])
    warp_size     = np.ceil(window_ratio*x.shape[2]).astype(int)
    window_steps  = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[2]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends   = (window_starts + warp_size).astype(int)
            
    output = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[1]):
            start_seg = pat[dim,:window_starts[i]]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[dim,window_starts[i]:window_ends[i]])
            end_seg = pat[dim,window_ends[i]:]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            output[i,dim,:] = np.interp(np.arange(x.shape[2]), np.linspace(0, x.shape[2]-1., num=warped.size), warped).T
    return output

# 24
def _window_warp_pos(x, pos_num=1, window_ratio=0.1, scales=[0.5, 2.]):
    
    pos_chnnl     = x.shape[1]//pos_num
    output = np.zeros_like(x)
    
    for i, pat in enumerate(x):
        for pos in range(pos_num):
            
            warp_scales   = np.random.choice(scales, x.shape[0])
            # window_ratio  = np.random.uniform(0.1, 0.5, 1)
            warp_size     = np.ceil(window_ratio*x.shape[2]).astype(int)
            window_steps  = np.arange(warp_size)
            window_starts = np.random.randint(low=1, high=x.shape[2]-warp_size-1, size=(x.shape[0])).astype(int)
            window_ends   = (window_starts + warp_size).astype(int)
            
            start_seg     = pat[pos*pos_chnnl:(pos+1)*pos_chnnl,:window_starts[i]]
            Interp_win    = interp1d(window_steps, pat[pos*pos_chnnl:(pos+1)*pos_chnnl,window_starts[i]:window_ends[i]], axis=-1)
            window_interp_steps = np.arange(0, window_steps[-1]+0.001, warp_scales[i])
            InterpVal_win = Interp_win(window_interp_steps)
            window_seg    = InterpVal_win
            end_seg       = pat[pos*pos_chnnl:(pos+1)*pos_chnnl,window_ends[i]:]
            warped        = np.concatenate((start_seg, window_seg, end_seg), axis=-1)
            
            Interp        = interp1d(np.arange(warped.shape[-1]), warped, axis=-1)
            interp_steps  = np.arange(0, warped.shape[-1]+0.001, (warped.shape[-1])/x.shape[-1])[:x.shape[-1]]
            if interp_steps[-1] > (warped.shape[-1]-1):
                interp_steps[-1] = (warped.shape[-1]-1)
            InterpVal     = Interp(interp_steps)
            output[i,pos*pos_chnnl:(pos+1)*pos_chnnl,:] = InterpVal
    
    return output

# 25
def _dimmagscale_chnnl(x, pos_num=1, magnitude_min=0, magnitude_max=0.4, M=5, split=10):
    '''This tfm applies magscale to each dimension independently'''
    magnitude = M*(magnitude_max-magnitude_min)/split
    if magnitude <= 0: return x
    
    scale = (1 + torch.rand((x.shape[0], x.shape[1], 1)) * magnitude)
    if np.random.rand() < .5: scale = 1 / scale
    output = x * scale.to(x.device)
    # output = output.reshape(x.shape[0], -1, x.shape[-1])
    return output

# 25
def _permutation_overall(x, pos_num=1, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[-1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    output = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[-1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            output[i] = pat[:,warp]
        else:
            output[i] = pat
    return output

# 26
def _magwarp_overall(x, pos_num=1, knot=4, magnitude=None):
    
    if magnitude==None:
        magnitude = 0.2
    else:
        magnitude = magnitude.data.to(x.device).numpy() * 0.4
    
    orig_steps = np.arange(x.shape[-1])
    
    random_warps = np.random.normal(loc=1.0, scale=magnitude, size=(x.shape[0], knot+2, x.shape[1]))
    warp_steps   = (np.ones((x.shape[1],1))*(np.linspace(0, x.shape[-1]-1., num=knot+2))).T
    output       = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper    = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[1])])
        output[i] = pat * warper

    return output

# 27
def _timewarp_chnnl(x, pos_num=1, sigma=0.2, knot=4):
    
    orig_steps   = np.arange(x.shape[-1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[1]))
    warp_steps   = (np.ones((x.shape[1],1))*(np.linspace(0, x.shape[-1]-1., num=knot+2))).T
    
    output       = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[1]):
            time_warp       = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale           = (x.shape[-1]-1)/time_warp[-1]
            output[i,dim,:] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[-1]-1), pat[dim,:])
    return output

# 28
def _tsw(sample, pos_num=1):
    
    data = sample.permute(0,2,1)
    data = data.detach().numpy()
    
    if data.shape[1] == 128 or data.shape[1] == 48:
        N = 8
    elif data.shape[1] == 200 or data.shape[1] == 500:
        N = 10
    elif data.shape[1] == 125:
        N = 5
    
    assert N < data.shape[1] and (data.shape[1] % N) == 0
    subsequences = np.split(data, N, axis=1)

    splits = np.arange(2, subsequences[0].shape[1], 2)

    # perform downsampling and upsampling alternately
    subsequences_aug = []
    for i in range(len(subsequences)):
        subsequence = subsequences[i]
        subsequence_splits = np.split(subsequence, splits, axis=1)
        
        # do not operate on the last split if its length is 1
        if subsequences[0].shape[1] % 2:
            tail = subsequence_splits[-1]
            subsequence_splits = subsequence_splits[:-1]

        if i % 2: # downsampling: average pooling with stride 2
            warped = list(map(lambda x: np.mean(x, axis=1), subsequence_splits))
            warped = np.stack(warped, axis=1)
        else: # upsampling: insert mean value between every two values with stride 2
            warped = list(map(lambda x: np.stack([x[:,0,:], np.mean(x, axis=1), x[:,1,:]], axis=1), subsequence_splits))
            warped = np.concatenate(warped, axis=1)
        
        if subsequences[0].shape[1] % 2:
            warped = np.concatenate([warped, tail], axis=1)

        subsequences_aug.append(warped)
    subsequences_aug = np.concatenate(subsequences_aug, axis=1)
    
    return torch.from_numpy(subsequences_aug).permute(0,2,1)

# 29 Mixup
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)    
        lam = max(lam, 1-lam)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup(batch_x, batch_y, classifier_obj, criterion, alpha, use_cuda=True):
    mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha, use_cuda=True)
    mixed_x, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
    output_bc = classifier_obj(mixed_x)[0]
    loss = mixup_criterion(criterion, output_bc, y_a, y_b, lam)
    return output_bc,loss

# 30 Cutmixup
def rand_bbox(size, lam_cut):
    
    W = size[2]
    cut_rat = np.sqrt(1. - lam_cut)
    cut_w = np.int(W * cut_rat)     
    # uniform
    cx = np.random.randint(W)
   
    #限制坐标区域不超过样本大小 
    # bbx1 = torch.from_numpy(np.clip(cx - cut_w // 2, 0, W)).int()   
    # bbx2 = torch.from_numpy(np.clip(cx + cut_w // 2, 0, W)).int()
    bbx1 = np.clip(cx - cut_w // 2, 0, W)    
    bbx2 = np.clip(cx + cut_w // 2, 0, W)    
    return bbx1, bbx2 #, bbx2, bby2

def cutmixup(input, target, criterion, model, lamda, use_cuda):
    # r = np.random.rand(1)
    new_input = input.clone()
    # generate mixed sample
    lam_cut = np.random.beta(lamda, lamda)
    lam_cut = max(lam_cut, 1-lam_cut)
    # lam_cut = np.concatenate([lam_cut[:,None], 1-lam_cut[:,None]], 1).max(1)
    lam_mix = np.random.beta(lamda, lamda) 
    # lam_mix = np.concatenate([lam_mix[:,None], 1-lam_mix[:,None]], 1).max(1)
    # lam_mix = input.new(lam_mix)
    lam_mix = lam_cut + (1-lam_cut)*lam_mix    
    lam_cut = lam_cut/lam_mix
    # lam_mix = input.new(lam_mix)
        
    if use_cuda:
        rand_index = torch.randperm(input.size()[0]).cuda()
    else:
        rand_index = torch.randperm(input.size()[0])
    target_a = target#一个batch
    target_b = target[rand_index] #batch中的某一张
    bbx1, bbx2 = rand_bbox(input.size(), lam_cut)
    input_shuffle = input[rand_index, :, :]
    new_input[:, :, bbx1:bbx2] = lam_mix * input[:, :, bbx1:bbx2] + (1-lam_mix) * input_shuffle[:, :, bbx1:bbx2]    
    lam = (1 - ((bbx2 - bbx1) / input.size()[-1])) * lam_mix
    # compute output
    output = model(new_input)[0]
    loss = criterion(output, target_a) * lam  + criterion(output, target_b) * (1. - lam)
    return output, loss

# 31 Cutmix
def rand_bbox(size, lam):
    
    W = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
     
    # uniform
    cx = np.random.randint(W)
   
    bbx1 = np.clip(cx - cut_w // 2, 0, W)    
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    
    return bbx1, bbx2 #, bbx2, bby2

def cutmix(input, target, criterion, model, lamda, use_cuda):
    # r = np.random.rand(1)
    new_input = input.clone()
    # generate mixed sample
    lam = np.random.beta(lamda, lamda)
    lam = max(lam, 1-lam)
    if use_cuda:
        rand_index = torch.randperm(input.size()[0]).cuda()
    else:
        rand_index = torch.randperm(input.size()[0])
    target_a = target#一个batch
    target_b = target[rand_index] #batch中的某一张
    bbx1, bbx2 = rand_bbox(input.size(), lam)
    new_input[:, :, bbx1:bbx2] = input[rand_index, :, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) / input.size()[-1])
    # compute output
    output = model(new_input)[0]
    loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    return output, loss

# 32 WDBA
# Core DTW
def _traceback(DTW, slope_constraint):
    i, j = np.array(DTW.shape) - 1
    p, q = [i-1], [j-1]
    
    if slope_constraint == "asymmetric":
        while (i > 1):
            tb = np.argmin((DTW[i-1, j], DTW[i-1, j-1], DTW[i-1, j-2]))

            if (tb == 0):
                i = i - 1
            elif (tb == 1):
                i = i - 1
                j = j - 1
            elif (tb == 2):
                i = i - 1
                j = j - 2

            p.insert(0, i-1)
            q.insert(0, j-1)
    elif slope_constraint == "symmetric":
        while (i > 1 or j > 1):
            tb = np.argmin((DTW[i-1, j-1], DTW[i-1, j], DTW[i, j-1]))

            if (tb == 0):
                i = i - 1
                j = j - 1
            elif (tb == 1):
                i = i - 1
            elif (tb == 2):
                j = j - 1

            p.insert(0, i-1)
            q.insert(0, j-1)
    else:
        sys.exit("Unknown slope constraint %s"%slope_constraint)
        
    return (np.array(p), np.array(q))

def dtw(prototype, sample, return_flag = 0, slope_constraint="asymmetric", window=None):
    """ Computes the DTW of two sequences.
    :param prototype: np array [0..b]
    :param sample: np array [0..t]
    :param extended: bool
    """
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"
    
    if window is None:
        window = s
    
    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i-window)
        end = min(s, i+window)+1
        cost[i,start:end]=np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    DTW = _cummulative_matrix(cost, slope_constraint, window)
        
    if return_flag == -1:
        return DTW[-1,-1], cost, DTW[1:,1:], _traceback(DTW, slope_constraint)
    elif return_flag == 1:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1,-1]

def _cummulative_matrix(cost, slope_constraint, window):
    p = cost.shape[0]
    s = cost.shape[1]
    
    # Note: DTW is one larger than cost and the original patterns
    DTW = np.full((p+1, s+1), np.inf)

    DTW[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        for i in range(1, p+1):
            if i <= window+1:
                DTW[i,1] = cost[i-1,0] + min(DTW[i-1,0], DTW[i-1,1])
            for j in range(max(2, i-window), min(s, i+window)+1):
                DTW[i,j] = cost[i-1,j-1] + min(DTW[i-1,j-2], DTW[i-1,j-1], DTW[i-1,j])
    elif slope_constraint == "symmetric":
        for i in range(1, p+1):
            for j in range(max(1, i-window), min(s, i+window)+1):
                DTW[i,j] = cost[i-1,j-1] + min(DTW[i-1,j-1], DTW[i,j-1], DTW[i-1,j])
    else:
        sys.exit("Unknown slope constraint %s"%slope_constraint)
        
    return DTW

# 33 WDBA
def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    
    x = x.squeeze(1).transpose(0,2,1)
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in range(ret.shape[0]):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw(prototype, sample, 0, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw(medoid_pattern, random_prototypes[nid], 1, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return np.expand_dims(ret.transpose(0,2,1),axis=1)

# 34 ShapeDTW
def shape_dtw(prototype, sample, return_flag = 0, slope_constraint="asymmetric", window=None, descr_ratio=0.05):
    """ Computes the shapeDTW of two sequences.
    :param prototype: np array [0..b]
    :param sample: np array [0..t]
    :param extended: bool
    """
    # shapeDTW
    # https://www.sciencedirect.com/science/article/pii/S0031320317303710
    
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"
    
    if window is None:
        window = s
        
    p_feature_len = np.clip(np.round(p * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s * descr_ratio), 5, 100).astype(int)
    
    # padding
    p_pad_front = (np.ceil(p_feature_len / 2.)).astype(int)
    p_pad_back = (np.floor(p_feature_len / 2.)).astype(int)
    s_pad_front = (np.ceil(s_feature_len / 2.)).astype(int)
    s_pad_back = (np.floor(s_feature_len / 2.)).astype(int)
    
    prototype_pad = np.pad(prototype, ((p_pad_front, p_pad_back), (0, 0)), mode="edge") 
    sample_pad = np.pad(sample, ((s_pad_front, s_pad_back), (0, 0)), mode="edge") 
    p_p = prototype_pad.shape[0]
    s_p = sample_pad.shape[0]
        
    cost = np.full((p, s), np.inf)
    for i in range(p):
        for j in range(max(0, i-window), min(s, i+window)):
            cost[i, j] = np.linalg.norm(sample_pad[j:j+s_feature_len] - prototype_pad[i:i+p_feature_len])
            
    DTW = _cummulative_matrix(cost, slope_constraint=slope_constraint, window=window)
    
    if return_flag == -1:
        return DTW[-1,-1], cost, DTW[1:,1:], _traceback(DTW, slope_constraint)
    elif return_flag == 1:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1,-1]

# 35 Window Slice
def window_slice(x, reduce_ratio=0.9):
    
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

# 36 DGW-sD
def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="shape", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    x = x.squeeze(1).transpose(0,2,1)
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(x):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]
        
        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]
                        
            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*shape_dtw(pos_prot, pos_samp, 0, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*shape_dtw(pos_prot, neg_samp, 0, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = shape_dtw(positive_prototypes[selected_id], pat, 1, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw(pos_prot, pos_samp, 0, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw(pos_prot, neg_samp, 0, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw(positive_prototypes[selected_id], pat, RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.9+0.1*warp_amount[i]/max_warp)[0]
    return np.expand_dims(ret.transpose(0,2,1),axis=1)

# 37 RGW-sD
def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="shape", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    x = x.squeeze(1).transpose(0,2,1)
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]
            
            if dtw_type == "shape":
                path = shape_dtw(random_prototype, pat, 1, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw(random_prototype, pat, 1, slope_constraint=slope_constraint, window=window)
                            
            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return np.expand_dims(ret.transpose(0,2,1),axis=1)

# 38 SFCC
def get_dft_coefs(X, n_coefs):
    """
    :param X: the training set  numpy array shape:[n_samples, n_time_steps]
    :param n_coefs: number of the coefficients
    :return: numpy array shape: [n_samples, n_coefs]
    """
    dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
                                norm_std=False)
    X_dft = dft.fit_transform(X)
    n_samples = len(X)
    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                            np.zeros((n_samples,))]
        ]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
        ]
    return X_dft_new

def sfcc(data, labels, N_classes, N_groups=4, aug_factor=1):

    data       = data.squeeze(1).transpose(0,2,1)
    N_channels = data.shape[-1]

    data_augmented   = []
    labels_augmented = []

    for i in range(N_classes):
        # stratify data by label
        index = np.where(labels == i)[0]
        if len(index) == 0:
            continue
        
        X = data[index]
        Y = labels[index]

        # get the time steps and the number of coefficients
        n_time_steps = len(X[0])
        
        # discrete fourier transform for each channel
        X_dft = []
        for i in range(N_channels):
            X_dft.append(get_dft_coefs(X[:,:,i], n_time_steps))
        X_dft = np.stack(X_dft, axis=2)

        # The splitting phase
        # split the X_dft to n groups for combination
        assert X_dft.shape[1] > N_groups
        split_X_dft = np.array_split(X_dft, N_groups, axis=1)

        # The random sampling phase ---
        # randomly samples the example in each split_data with tha corresponding label
        # the number of the sampled_data are half the original size of dataset
        sampled_X_dft = []
        for j in range(N_groups):
            split_data = split_X_dft[j]
            n_samples = split_data.shape[0]
            N_selected = int(n_samples * aug_factor)
            N_list = np.arange(n_samples)
            # start to sample
            selected_idx = np.random.choice(N_list, N_selected, replace=False)
            sampled_X_dft.append(split_data[selected_idx])

        # concatenate the sampled data from the n_group
        sampled_X_dft = np.concatenate(sampled_X_dft, axis=1)

        # inverse the frequency domain into time domain with the same time step
        X_combined = []
        for i in range(N_channels):
            X_combined.append(np.fft.irfft(sampled_X_dft[:,:,i], n_time_steps))
        X_combined = np.stack(X_combined, axis=2)
        
        data_augmented.append(X_combined)
        labels_augmented.append(Y)

    data_augmented = np.concatenate(data_augmented)
    labels_augmented = np.concatenate(labels_augmented)

    return np.expand_dims(data_augmented.transpose(0,2,1), axis=1), labels_augmented

dispatcher = { 'MagNoise' : _magnoise, 'Magscale' : _dimmagscale, 'MagWarpChnnl': _magwarp, 'Timewarp': _timewarp, 
               'Timenoise':_timenoise, 'RandomZoom': _randomzoom, 'TimeStepZero': _timestepzero, 'Scaling': _scaling,
               'Cutout':_cutout, 'Timecrop':_randomcrop, 'Permutation':_permutation, 'Resample':_resample,
               'GenerateHigh':_generate_high, 'GenerateLow':_generate_low, 'AmpNoise':_ifft_amp_noise,
               'PhaseShift':_ifft_phase_shift, 'AmpPhaseShiftWin':_ifft_amp_phase_shift_win, 'AmpPhaseShiftFully':_ifft_amp_phase_shift_fully,
               'RandRotate': _randrotate, 'TimeShift':_timeshift, 'WindowSliceOverall': _window_slice_overall, 'WindowSlicePos': _window_slice_pos,
               'WindowWarpOverall': _window_warp_overall, 'WindowWarpPos': _window_warp_pos, 'DimmagscaleChnnl': _dimmagscale_chnnl,
               'PermutationOverall': _permutation_overall, 'Magwarp': _magwarp_overall, 'TimewarpChnnl': _timewarp_chnnl, 'TSW': _tsw,  'MagNoise' : _magnoise, 
               'Ampperturb': _ifft_amp_shift, 'Phaseperturb':_ifft_phase_shift, 'AmpPhaseShiftWin':_ifft_amp_phase_shift_win, 'AmpPhaseShiftFully':_ifft_amp_phase_shift_fully}

# class DataAugment(Data.Dataset):
#     def  __init__(self, x, y, aug_methods, pos_num):
#         x_torch = torch.FloatTensor(x)
#         y_torch = torch.tensor(y).long()
#         self.data  = x_torch
#         self.label = y_torch
#         self.aug_methods = aug_methods
#         self.pos_num     = pos_num

#     def __getitem__(self, index):
#         data_x = self.data[index]
#         for aug_method in self.aug_methods:
#             if dispatcher.get(aug_method) is not None:
#                 data_x = dispatcher[aug_method](data_x, pos_num=self.pos_num)
#         data = (data_x, self.label[index])
#         return data

#     def __len__(self):
#         return self.data.shape[0]

# # DifDataAugment
# class DifDataAugment(Data.Dataset):
    
#     def  __init__(self, x, y, pos_num, magnitudes, policies, search_flag=True):
#         x_torch    = torch.FloatTensor(x)
#         y_torch    = torch.tensor(y).long()
#         self.data  = x_torch
#         self.label = y_torch
#         self.pos_num      = pos_num
#         self.magnitudes   = magnitudes
#         self.policies     = policies
#         self.search_flag  = search_flag

#     def __getitem__(self, index):
        
#         data_x              = self.data[index]
#         if self.search_flag:
#             magnitudes          = self.magnitudes.clamp(0, 1)
#             probabilities_index = self.probabilities_index
#             for i, sub_policy in enumerate(self.policies):
#                 for j, aug_method in enumerate(sub_policy):
#                     if probabilities_index[i][j].item() == 1:
#                         if dispatcher.get(aug_method) is not None:
#                             data_x = dispatcher[aug_method](torch.FloatTensor(data_x), pos_num=self.pos_num, magnitude=magnitudes[i][j])
#         data = (data_x, self.label[index])
        
#         return data

#     def __len__(self):
#         return self.data.shape[0]

# class DifferentiableAugment(nn.Module):
#     def __init__(self, sub_policy):
#         super(DifferentiableAugment, self).__init__()
#         self.sub_policy = sub_policy

#     def forward(self, origin_data, probability, probability_index, magnitude):
#         index    = sum( p_i.item()<<i for i, p_i in enumerate(probability_index))
#         com_data = 0
#         data     = origin_data
#         adds     = 0

#         for selection in range(2**len(self.sub_policy)):
#             trans_probability = 1
#             for i in range(len(self.sub_policy)):
#                 if selection & (1<<i):
#                     trans_probability = trans_probability * probability[i]
#                     if selection == index:
#                         data = data - magnitude[i]
#                         adds = adds + magnitude[i]
#                 else:
#                     trans_probability = trans_probability * ( 1 - probability[i] )
#             if selection == index:
#                 data = data.detach() + adds
#                 com_data = com_data + trans_probability * data
#             else:
#                 com_data = com_data + trans_probability

#         return com_data

# class MixedAugment(nn.Module):
#     def __init__(self, sub_policies):
#         super(MixedAugment, self).__init__()
#         self.sub_policies = sub_policies
#         self._compile(sub_policies)

#     def _compile(self, sub_polices):
#         self._ops = nn.ModuleList()
#         self._nums = len(sub_polices)
#         for sub_policy in sub_polices:
#             ops = DifferentiableAugment(sub_policy)
#             self._ops.append(ops)

#     def forward(self, origin_data, probabilities, probabilities_index, magnitudes, weights=torch.tensor([1,1])):
#         return sum(w * op(origin_data, p, p_i, m)
#                    for i, (p, p_i, m, w, op) in
#                    enumerate(zip(probabilities, probabilities_index, magnitudes, weights, self._ops)))/2

# def _concat(xs):
#     return torch.cat([x.contiguous().view(-1) for x in xs])

# class Architect(object):

#     def __init__(self, model, args):
#         self.network_momentum = args.momentum
#         self.network_weight_decay = args.weight_decay
#         self.model = model
#         self.optimizer = torch.optim.Adam(self.model.augment_parameters(),
#                                           lr=args.arch_learning_rate, betas=(0.5, 0.999),
#                                           weight_decay=args.arch_weight_decay)
#         self.scheduler_arch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.5)

#     def _compute_unrolled_model(self, input, target, eta, network_optimizer):
#         loss   = self._loss_function(self.model(input, test_flag=True)[0], target) # compute train loss
#         theta  = _concat(self.model.param_list).data.detach()
#         moment = torch.zeros_like(theta)
#         dtheta         = _concat(torch.autograd.grad(loss, self.model.param_list)).data.detach() + self.network_weight_decay * theta
#         theta_mid      = theta - eta * (moment + dtheta)              # calculate w'，corrsponding to w' = w − ζ▽wE[Ltrain(w; d)]
#         unrolled_model = self._construct_model_from_theta(theta_mid)  # obtain w'-based model，which is used to calculate ▽w'Lval(w')
#         return unrolled_model

#     def step(self, input_train, target_train, input_valid, target_valid, eta, loss_function, network_optimizer, unrolled): ## eta: learning rate
#         self._loss_function = loss_function
#         self.optimizer.zero_grad()   ## update DA policy
#         if unrolled:
#             self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
#         else:
#             self._backward_step(input_valid, target_valid)
#         self.optimizer.step()

#     def _backward_step(self, input_valid, target_valid):
#         loss = self._loss_function(self.model(input_valid, test_flag=True)[0], target_valid)
#         loss.backward(retain_graph=True)

#     def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
#         unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
#         unrolled_model.set_augmenting(False)                             # now d is fixed
#         unrolled_loss  = self._loss_function(unrolled_model(input_valid, test_flag=True)[0], target_valid) # calcualte Lval(w')

#         unrolled_loss.backward()
#         dalpha = []
#         vector = [v.grad.data.detach() for v in unrolled_model.parameters()] # calculate ▽w'Lval(w')
#         implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
#         for ig in implicit_grads:
#             if ig is None:
#                 dalpha += [None]
#             else:
#                 dalpha += [-ig]

#         for v, g in zip(self.model.augment_parameters(), dalpha):
#             if v.grad is None:
#                 if not (g is None):
#                     v.grad = Variable(g.data)
#             else:
#                 if not (g is None):
#                     v.grad.data.copy_(g.data)                   # update the learnalble augmentation parameters d

#     def _construct_model_from_theta(self, theta):
#         model_new  = self.model.new()
#         del model_new._modules['conv_sigma']
        
#         model_dict = self.model.state_dict()
#         del model_dict['conv_sigma.0.weight']
#         del model_dict['conv_sigma.0.bias']
#         del model_dict['conv_sigma.1.weight']
#         del model_dict['conv_sigma.1.bias']
#         del model_dict['conv_sigma.1.running_mean']
#         del model_dict['conv_sigma.1.running_var']
#         del model_dict['conv_sigma.1.num_batches_tracked']
        

#         params, offset = {}, 0
#         for k, v in self.model.named_parameters():
#             if k != 'conv_sigma.0.weight' and k != 'conv_sigma.0.bias' and k != 'conv_sigma.1.weight' and k != 'conv_sigma.1.bias' and k != 'conv_sigma.1.running_mean' and k != 'conv_sigma.1.running_var' and k != 'conv_sigma.1.num_batches_tracked':
#                 v_length = np.prod(v.size()) # 16*3*3*3，calculate the parameter number of v
#                 params[k] = theta[offset: offset + v_length].view(v.size())
#                 offset = offset + v_length

#         assert offset == len(theta)
#         model_dict.update(params)
#         model_new.load_state_dict(model_dict)
#         return model_new.cuda()

#     def _hessian_vector_product(self, vector, input, target, r=1e-2):
#         R = r / _concat(vector).data.detach().norm()  ## small scalar in original paper，corresponding to ε=0.01/||▽w'Lval(w')||2
#         # for p, v in zip(self.model.parameters(), vector):
#         for p, v in zip(self.model.param_list, vector):
#             p.data.add_(R, v) # p += R * v            # calculate w+ = w + ε▽w'Lval(w')
#             # p.data = p.data + R * v
#         loss    = self._loss_function(self.model(input, test_flag=True)[0], target)
#         grads_p = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)   # equation 19 left，calculate ▽dE[Ltrain(w+, d)]
#         # print(grads_p)

#         # for p, v in zip(self.model.parameters(), vector):
#         for p, v in zip(self.model.param_list, vector):
#             p.data.sub_(2 * R, v) # p -= 2*R * v，calculate w- = w - ε▽w'Lval(w')
#             # p.data = p.data - 2*R * v
#         loss = self._loss_function(self.model(input, test_flag=True)[0], target)
#         grads_n = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)   ## equation 19 right
#         # print(grads_n)

#         for p, v in zip(self.model.param_list, vector):
#             p.data.add_(R, v)                             # restitution

#         return [ None if ( x is None ) or ( y is None) else (x - y).div_(2 * R) for x, y in zip(grads_p, grads_n) ]  ## equation 19