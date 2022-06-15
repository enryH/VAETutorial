# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # Typical VAEs
# This notebook is a walk-through of the code to produce the results in Section 6. First, let us see what our ground truth datasets look like. 
#
# - [repo](https://github.com/ronaldiscool/VAETutorial)
#
#   - original notebook in <a href="https://colab.research.google.com/github/ronaldiscool/VAETutorial/blob/master/VAEtoy2d.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#   - forked notebook in 

# %%
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


# %% [markdown]
# ### Grid layout
#
# - data is a set of 2D vectors with a certain distribution
# - grid gives within the specified limits a grid of points

# %%
from itertools import product

def get_grid(limit: int, step: int, dim=2):
    # grid = np.array([[a, b]
    #                  for a in np.arange(-limit, limit, step)
    #                  for b in np.arange(-limit, limit, step)])
                     
    grid = np.array(list(product(np.arange(-limit, limit, step), repeat=dim)))
    # maybe meshgrid?
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    return grid


get_grid(1.0, 1/2)


# %%
def get_centers():
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
               (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)),
               (-1. / np.sqrt(2), 1. / np.sqrt(2)),
               (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    return centers


get_centers()


# %%
def gaussian_data_8centers(limit=0.5, step=1/1024):
    vectors = int(2*limit/step)
    grid = get_grid(limit=limit, step=step)
    centers = get_centers()
    centers = [(limit * x, limit * y) for x, y in centers]
    color = np.zeros((vectors*vectors, 3))
    for i, center in enumerate(centers):
        # value * std(?) - center shifting values
        x = grid*1.414 - center
        print(f"For {grid[0] = }, transformed data is {x[0] = }")
        # prob of each data point, likelihood for each row
        prob = np.prod(1/(2*np.pi/256.0)**0.5 * np.exp(-x**2/(2/256.0)), -1)
        color[:, 0] += i/8.0 * prob
        color[:, 2] += prob
    return color


color = gaussian_data_8centers(limit=1, step=1/2)
print("likelihood for each row, accumulated for 8 gaussians:")
print(color[:,2])
print("color, each row: center/8 * prob, zeros, prob")
print(color)


# %%
for coordinates, p in zip(get_grid(limit=1, step=1/2), color[:,2]): print(f"coordinate {str(coordinates):>13}, prob: {p :>12.8f}")


# %%
def checkerboard_8colors(limit=0.5, step=1/1024):
    pixels = int(2*limit/step)
    logging.debug(f"{pixels}")
    grid = get_grid(limit=limit, step=step)
    l = [0, 2, 1, 3, 0, 2, 1, 3]
    color = np.zeros((pixels, pixels, 3))
    stride = pixels // 4
    logging.debug(f"{stride = }")
    for i in range(8):
        y = i//2*stride
        logging.debug(f"{y = }")
        x = l[i]*stride
        logging.debug(f"{x = }")
        color[x:x+stride, y:y+stride, 0] = i/8.0
        color[x:x+stride, y:y+stride, 2] = 1
    return color


LIMIT = 0.5
color = checkerboard_8colors(limit=LIMIT, step=1/16)

# print("color, each row: center/8 * prob, zeros, prob")
# print(color.shape, color, sep="\n")

fig, ax = plt.subplots(figsize=(18, 18))
ax.axis('off')
ax = ax.imshow(color)  # , extent=(-LIMIT, LIMIT, -LIMIT, LIMIT))


# %%
def spirals(limit=0.5, **kwargs):
    step = 1/1024
    pixels = int(2*limit/step)
    logging.debug(f"{pixels}")
    grid = get_grid(limit=limit, step=step)
    grid = grid.reshape((pixels, pixels, 2))
    color = np.zeros((pixels, pixels, 3))
    for i in range(10000):
        n = (i/10000.0)**0.5 * 540 * 2 * np.pi / 360
        d = np.zeros((1, 2))
        d[0, 0] = -np.cos(n) * n/3.0/8
        d[0, 1] = np.sin(n) * n / 3.0 / 8

        idx = int((d[0, 0]+limit)/step)
        idy = int((d[0, 1]+limit)/step)
        x = grid[idx-50:idx+50, idy-50:idy+50, :] - d
        cur_prob = np.prod(1/(2*np.pi*0.01/64)**0.5 *
                           np.exp(-x**2/(2*0.01/64)), -1)

        cur_color = np.ones((100, 100, 3))
        cur_color[:, :, 0] = (i/20000.0+0.5)*cur_prob
        cur_color[:, :, 2] = cur_prob
        color[idx-50:idx+50, idy-50:idy+50] += cur_color

        # other spiral
        idx = int((-d[0, 0]+limit)/step)
        idy = int((-d[0, 1]+limit)/step)
        x = grid[idx-50:idx+50, idy-50:idy+50, :] + d
        cur_prob = np.prod(1/(2*np.pi*0.01/64)**0.5 *
                           np.exp(-x**2/(2*0.01/64)), -1)

        cur_color = np.ones((100, 100, 3))
        cur_color[:, :, 0] = (-i/20000.0+0.5)*cur_prob
        cur_color[:, :, 2] = cur_prob
        color[idx-50:idx+50, idy-50:idy+50] += cur_color
    return color


color = spirals(limit=0.5)
print(f"{color.shape = }")

fig, ax = plt.subplots(figsize=(18, 18))
ax.axis('off')
ax = ax.imshow(color)


# %%
#ENUM or dict for functions
data_generating_fcts = {
    '8gaussians' : gaussian_data_8centers,
    'checkerboard' : checkerboard_8colors,
    '2spirals': spirals
}

def plot_gt(data):
    limit = 0.5
    step = 1/1024.0
    pixels = int(2*limit/step) # 1024
    logging.debug(f"Number of pixels: {pixels}x{pixels}")
    grid = get_grid(limit=limit, step=step)

    color = data_generating_fcts[data](limit=limit, step=step)

    color = color.reshape((pixels,pixels,3)) # partly already in shape
    color[:,:,0]/=(color[:,:,2]+1e-12) # set zero channel also to prob
    color[:,:,1]=1 # first channel: set to 1
    prob = color[:,:,2].reshape((pixels,pixels))
    # entropy calc.
    prob = prob / np.sum(prob) #normalize the data (now it is garuanteed prob)
    
    prob+=1e-20 # stablize small values
    entropy = - prob * np.log(prob)/np.log(2)
    entropy = np.sum(entropy)
    
    max_prob = np.max(prob)

    color[:,:,2]/=np.max(color[:,:,2])  # normalize by maximum prob
    color[:,:,1]=color[:,:,2] # now also set the frist channel to last channel
    color = np.clip(color, 0, 1) # constrain values between zero and one
    color = col.hsv_to_rgb(color) # hue saturation lightness -> to rgb


    fig = plt.figure(figsize=(18, 18))

    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.imshow(prob, extent=(-limit, limit, -limit, limit))

    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.imshow(color, extent=(-limit, limit, -limit, limit))

    fig.tight_layout()

    return entropy-20, max_prob, prob, color

entropy8g, max_prob8g, prob8g, color8g = plot_gt('8gaussians')
print('Entropy for 8gaussians: {:f}'.format( entropy8g))
#print('Max probability for 8gaussians: {:e}'.format(max_prob8g))

entropyc, max_probc, probc, colorc =plot_gt('checkerboard')
print('Entropy for Checkerboard: {:f}'.format( entropyc))
#print('Max probability for Checkerboard: {:e}'.format(max_probc))

entropy2s, max_prob2s, prob2s, color2s =plot_gt('2spirals')
print('Entropy for 2spirals: {:f}'.format( entropy2s))
#print('Max probability for 2spirals: {:e}'.format(max_prob2s))



# %%
print(f"{max_prob8g = :.8f}")
print(f"{max_probc  = :.8f}")
print(f"{max_prob2s = :.8f}")


# %% [markdown] id="C6B7eXoWMvz9"
# We have plotted two figures for each dataset.
# On the left-hand side is the ground truth density function. On the right-hand side is a color map of the data that we will later use to visualize the latent space of a VAE. 
#
# We have also printed out the approximate entropy *H* of the ground truth probability distribution over the pixels. Recall that for continuous data, *H + 2n* can be interpreted as the number of bits needed to describe a sample from a 2D distribution to *n*-bit accuracy. As a sanity check, a uniform distribution across the whole domain should have an entropy of 0 bits, and the checkerboard dataset, which is a uniform distribution over half the domain, would have an entropy of -1. The entropy is also the value of the optimal negative log-likelihood for a maximum likelihood model.
#
# Now let us set up the dataloader. We will call ```sample2d``` every iteration during training to sample a batch of continuous values from the ground truth dataset to train the VAE.

# %%
# logging.getLogger().setLevel(logging.DEBUG)
def batch_8gaussians(batch_size=200, scale=4, seed=None):
    """get one batch for 8 gaussians

    Parameters
    ----------
    batch_size : int, optional
        _description_, by default 200
    scale : int, optional
        scale the means of the centers, by default 4

    Returns
    -------
    _type_
        _description_
    """
    
    rng = np.random.RandomState(seed=seed)
    
    centers = get_centers()
    centers = [(scale * x, scale * y) for x, y in centers]
    logging.debug(f"{centers = }")
    dataset = []
    #dataset = np.zeros((batch_size, 2))
    for i in range(batch_size):
        # fix this to be faster by using a dataset
        point = rng.randn(2) * 0.5
        logging.debug(f"{point = }")
        idx = rng.randint(8)
        center = centers[idx]
        point += center
        # point[0] += center[0]
        # point[1] += center[1]
        dataset.append(point)
        #dataset[i]=point
    dataset = np.array(dataset, dtype='float32')
    dataset /= 1.414
    return dataset/8.0

batch_8gaussians(10, scale=4, seed=42)

# %%
import torch
from torch.utils.data import DataLoader, Dataset

class Dataset8Gaussians(Dataset):

    def __init__(self, seed=None, N=60_000):
        self.centers = get_centers()
        self.seed = seed
        self.rng = rng = np.random.default_rng(seed=seed)
        self.N = 60_000

    def __getitem__(self, i):
        point = self.rng.standard_normal(2, dtype=np.float32) * 0.5 # draw from std. normal dist and rescale
        logging.debug(f"{point = }")
        idx = self.rng.integers(8) # select a random center from the 8 guassians
        center = self.centers[idx] 
        point += center # place random point w.r.t. selected center (-> drawn from center)
        point /= (1.414*8.0) # normalize by this constant -> WHY?
        point = torch.from_numpy(point)
        return point

    def __len__(self):
        return self.N

logging.getLogger().setLevel(logging.DEBUG)
dataset_8gaussians = Dataset8Gaussians()
dataset_8gaussians[5]

# %%
logging.getLogger().setLevel(logging.INFO)
dl_8guassians = DataLoader(dataset=dataset_8gaussians, batch_size=8, shuffle=True, num_workers=0)
next(iter(dl_8guassians))


# %%
def sample2d(data, batch_size=200):
    #code largely taken from https://github.com/nicola-decao/BNAF/blob/master/data/generate2d.py

    rng = np.random.RandomState()

    if data == '8gaussians':
        scale = 4
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        #dataset = np.zeros((batch_size, 2))
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
            #dataset[i]=point
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414
        return dataset/8.0

    elif data == '2spirals':
        n = np.sqrt(np.random.rand(batch_size, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n
        d1y = np.sin(n) * n 
        x = np.hstack((d1x, d1y)) / 3 * (np.random.randint(0, 2, (batch_size,1)) * 2 -1)
        x += np.random.randn(*x.shape) * 0.1
        return x/8.0

    elif data == 'checkerboard':
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2 / 8.0

    else:
        raise RuntimeError
sample2d('8gaussians', batch_size=2)

# %% [markdown] id="21LuJO5yMuH-"
# Now that we have set up our data loader, we can construct a Typical VAE. The architectural backbone of our encoder and decoder will be a *DenseBlock*, which concatenates the output of each fully connected layer with its input.

# %%
from torch import nn

class DenseBlock(nn.Module):
  def __init__(self, input_dim, growth, depth):
    super(DenseBlock,self).__init__()
    ops=[]
    for i in range(depth):
      ops.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim+i*growth, growth)), nn.ReLU() ) )

    self.ops = nn.ModuleList(ops)

  def forward(self,x):
    for op in self.ops:
      y = op(x)
      x = torch.cat([x,y],1)
    return x
    
DenseBlock(2, 4, 2), DenseBlock(100, 32, 3) # which concatenates the output of each fully connected layer with its input

# %% [markdown] id="T2l0BFnupIxc"
# Note that we use Weight Norm as opposed to Batch Norm. The reasononing behind using Weight Norm is that Batch Norm introduces noise during training, which although tolerable for classification hurts our ability to precisely reconstruct the input. By using Weight Norm, any noise introduced in our VAE is counted towards the regularization loss. We can now construct our VAE architecture and define functions to perform inference. ```compute_negative_elbo``` computes the negative ELBO and will be used during training. ```importance_sampling``` approximates the exact negative log-likelihood and will be used during evaluation.
#
# During training, we use a trick called the *free bits objective*. A problem with optimizing VAEs is that the latent space initially contains 0 information, so a lot of noise is applied to the space. This noise can send the optimization landscape into a bad local minima and cause swings in behavior depending on the random seed. To stabilize training, the free bits objective introduces a hyperparameter $\alpha$ such that the regularization loss is inactive for latent variables containing less than $\alpha$ bits of information. This allows the VAE to quickly learn to store $\alpha$ bits into each latent variable, after which training can stablely proceed with the standard negative ELBO objective. We want $\alpha$ to be high enough so that the VAE enters a stable regime of training but do not want $\alpha$ to be too high since the free bits objective is not the natural VAE objective. Moreover, if $\alpha$ is too high, then $q_\phi(\mathbf{z}|\mathbf{x})$ will very quickly have low variance, which may prevent exploration of the latent space. In our example we will use $\alpha=0.05$.

# %%
import torch
import math
import torch.nn.functional as F

class VAE(nn.Module):
  def __init__(self, 
               latent_dim=2,
               growth=1024,
               depth=6):
    super(VAE,self).__init__()
    #set up hyperparameters
    self.latent_dim = latent_dim

    #define architecture
    encoder_dense = DenseBlock(2, growth,depth) 
    encoder_linear = nn.utils.weight_norm(nn.Linear(2+growth*depth, self.latent_dim*2))
    self.encoder = nn.Sequential(encoder_dense, encoder_linear)

    decoder_dense = DenseBlock(self.latent_dim, growth,depth) 
    decoder_linear = nn.utils.weight_norm(nn.Linear(self.latent_dim+growth*depth, 2*2))
    self.decoder = nn.Sequential(decoder_dense, decoder_linear)

  def encode(self,x):
    z_params = self.encoder(x)
    z_mu = z_params[:,:self.latent_dim]
    z_logvar = z_params[:,self.latent_dim:]
    return z_mu, z_logvar

  def decode(self,z):
    x_params = self.decoder(z)
    x_mu = x_params[:,:2]
    x_logvar = x_params[:,2:]
    return x_mu, x_logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    return mu + torch.cuda.FloatTensor(std.shape).normal_() * std

  def forward(self,x):
    z_mu, z_logvar = self.encode(x)
    z = self.reparameterize(z_mu, z_logvar)
    x_mu, x_logvar = self.decode(z)
    return x_mu,x_logvar, z_mu, z_logvar


  def compute_negative_elbo(self, x, freebits=0):
    x_mu, x_logvar, z_mu, z_logvar = self.forward(x)
    l_rec = -torch.sum(gaussian_log_prob(x, x_mu, x_logvar),1)
    l_reg = torch.sum(F.relu(self.compute_kld(z_mu, z_logvar)-freebits*math.log(2))+freebits*math.log(2),1)
    return l_rec + l_reg, l_rec, l_reg

  def compute_kld(self,z_mu, z_logvar):
    return 0.5*(z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)

  
  def importance_sampling(self, x, importance_samples=1):
    z_mu, z_logvar = self.encode(x)

    z_mu = z_mu.unsqueeze(1).repeat((1,importance_samples,1))
    z_mu = z_mu.reshape((-1, self.latent_dim))
    z_logvar = z_logvar.unsqueeze(1).repeat((1,importance_samples,1))
    z_logvar = z_logvar.reshape((-1, self.latent_dim))
    x = x.unsqueeze(1).repeat((1,importance_samples,1))
    x = x.reshape((-1,2))

    z = self.reparameterize(z_mu, z_logvar)
    x_mu, x_logvar = self.decode(z)
    x_mu = x_mu.reshape((-1,importance_samples,2))
    x_logvar = x_logvar.reshape((-1,importance_samples,2))

    x = x.reshape((-1,importance_samples,2))
    z = z.reshape((-1,importance_samples,self.latent_dim))
    z_mu = z_mu.reshape((-1,importance_samples,self.latent_dim))
    z_logvar = z_logvar.reshape((-1,importance_samples,self.latent_dim))

    logpxz = gaussian_log_prob(x, x_mu,x_logvar)
    logpz = gaussian_log_prob(z, torch.zeros(z.shape).cuda(), torch.zeros(z.shape).cuda())
    logqzx = gaussian_log_prob(z, z_mu, z_logvar)

    logprob = logpxz+logpz - logqzx
    logprob = torch.sum(logprob,2)
    logprob = log_mean_exp(logprob, 1)

    return -logprob

def gaussian_log_prob(z, mu, logvar):
  return -0.5*(math.log(2*math.pi) + logvar + (z-mu)**2/torch.exp(logvar))

def log_mean_exp(x,axis):
    m,_=torch.max(x,axis)
    m2,_ = torch.max(x,axis,keepdim=True)
    return m + torch.log(torch.mean(torch.exp(x-m2),axis))



# %%
VAE(2, 5, 3) # input and output of layer -> new input for next layer

# %% [markdown] id="glz3iGicAwWh"
# We now have all the tools we need to train a VAE. We will sample 200 points each iteration and minimize the negative ELBO over the course of 60,000 iterations. Each model should take roughly 20 minutes to train.

# %%
import time

def s2hms(s):
  h = s//3600
  m = (s-h*3600)//60
  s = int(s-h*3600-m*60)
  return h,m,s

def print_progress(time, cur_iter, total_iter):
  h,m,s = s2hms(time)
  h2,m2,s2 = s2hms(time*total_iter/cur_iter - time)
  print('Time Elapsed: %d hours %d minutes %d seconds. Time Remaining: %d hours %d minutes %d seconds.'%(h,m,s,h2,m2,s2))


def train_vae(dataset, model=None, epochs=60000, print_freq=1000, batch_size=200):
  if model is None:
    model = VAE().cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                          patience=epochs//20,
                                                          min_lr=1e-8, verbose=True,
                                                          threshold_mode='abs')
  start=time.time()
  loss_ema=0
  best_ema =1e9
  for iteration in range(epochs): #train for 60k iterations
    data = torch.tensor(sample2d(dataset,batch_size)).float().cuda()
    # if iteration==0: print(data)
    neg_elbo, l_rec, l_reg = model.compute_negative_elbo(data,freebits=0.05)

    loss = torch.mean(l_reg + l_rec )/math.log(2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_ema = 0.999*loss_ema + 0.001*loss
    data=None
    #scheduler.step(loss_ema)
    if iteration == int(epochs*0.6)  or iteration == int(epochs*0.7) or iteration == int(epochs*0.8) or iteration == int(epochs*0.9):
       for param_group in optimizer.param_groups:
           param_group['lr'] /= 2
    if iteration%print_freq == 0:
      with torch.no_grad():
        #print('Iteration %d. Loss: %f'%(iteration, loss))
        #neg_elbo, l_rec, l_reg = model.compute_negative_elbo(data,0)
        print('Iteration %d. EMA: %f ELBO: %f L_rec: %f L_reg: %f'%(iteration, loss_ema, torch.mean(neg_elbo)/math.log(2),torch.mean(l_rec)/math.log(2), torch.mean(l_reg)/math.log(2)))
        print_progress(time.time()-start, iteration+1, epochs)

  return model, optimizer



# %%
fpath_model8g= '8gaussians.pt'
if Path(fpath_model8g).exists():
  checkpoint = torch.load(fpath_model8g)
  model8g = VAE().cuda()
  opt8g = torch.optim.Adam(model8g.parameters(), lr=1e-4, amsgrad=True)

  model8g.load_state_dict(checkpoint['model_state_dict'])
  opt8g.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    model8g, opt8g = train_vae('8gaussians',None)
    fpath_model8g= '8gaussians.pt'
    torch.save({'model_state_dict': model8g.state_dict(),
            'optimizer_state_dict': opt8g.state_dict()
            },
            fpath_model8g
            )

# %%
# DataLoader implementation
# def train_vae(dl:DataLoader, model=None, epochs=60000, print_freq=1000, optimizer=None):
#   if model is None:
#     model = VAE().cuda()
#   if optimizer is None:
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
#   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
#                                                           patience=epochs//20,
#                                                           min_lr=1e-8, verbose=True,
#                                                           threshold_mode='abs')
#   start=time.time()
#   loss_ema=0
#   best_ema =1e9
#   for iteration in range(epochs): #train for 60k iterations
#     for data in dl:
#       # if iteration==0: print(data)
#       data = data.cuda()
#       neg_elbo, l_rec, l_reg = model.compute_negative_elbo(data,freebits=0.05)

#       loss = torch.mean(l_reg + l_rec )/math.log(2)
#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()
#       loss_ema = 0.999*loss_ema + 0.001*loss
#       data=None
#     #scheduler.step(loss_ema)
#     if iteration == int(epochs*0.6)  or iteration == int(epochs*0.7) or iteration == int(epochs*0.8) or iteration == int(epochs*0.9):
#        for param_group in optimizer.param_groups:
#            param_group['lr'] /= 2
#     if iteration%print_freq == 0:
#       with torch.no_grad():
#         #print('Iteration %d. Loss: %f'%(iteration, loss))
#         #neg_elbo, l_rec, l_reg = model.compute_negative_elbo(data,0)
#         print('Iteration %d. EMA: %f ELBO: %f L_rec: %f L_reg: %f'%(iteration, loss_ema, torch.mean(neg_elbo)/math.log(2),torch.mean(l_rec)/math.log(2), torch.mean(l_reg)/math.log(2)))
#         print_progress(time.time()-start, iteration+1, epochs)

#   return model, optimizer

# dataset = Dataset8Gaussians()
# dl = DataLoader(dataset, num_workers=0, shuffle=True, batch_size=1000)

# fpath_model8g= '8gaussians.pt'
# if Path(fpath_model8g).exists():
#   checkpoint = torch.load(fpath_model8g)
#   model8g = VAE().cuda()
#   opt8g = torch.optim.Adam(model8g.parameters(), lr=1e-4, amsgrad=True)

#   model8g.load_state_dict(checkpoint['model_state_dict'])
#   opt8g.load_state_dict(checkpoint['optimizer_state_dict'])


# # logging.getLogger().setLevel(logging.INFO)
# # model8g, opt8g = train_vae(dl, model=None, epochs=2, print_freq=1)
# model8g, opt8g = train_vae(dl, model8g, 1000, 10, optimizer=opt8g) #should take ~20 minutes to train

# %%
modelc = train_vae('checkerboard',None,60000,100) #should take ~20 minutes to train

# %%
model2s = train_vae('2spirals',None,60000,100) #should take ~20 minutes to train


# %% [markdown]
# We see that the reconstruction loss decreases as training progresses while the regularization loss increases.
# Now that we have trained a model, we can evaluate it by comparing the negative log-likelihood with the ground-truth distribution and by plotting the density map of the model. The negative log-likelihood is approximated with importance sampling.

# %%
def plot_density2d(model, max_prob, gt_prob, importance_samples=1):
    #code largely taken from https://github.com/nicola-decao/BNAF/blob/master/toy2d.py
    limit=0.5
    step=1/1024.0
    grid = torch.Tensor([[a, b] for a in np.arange(-limit, limit, step) for b in np.arange(-limit, limit, step)])
    grid_dataset = torch.utils.data.TensorDataset(grid.cuda())
    grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=20000//importance_samples, shuffle=False)

    l=[]
    start=time.time()
    with torch.no_grad():
      for idx, (x_mb,) in enumerate(grid_data_loader):
        temp= model.importance_sampling(x_mb,importance_samples)
        l.append(torch.exp(-temp))
        if idx % 600 == 0 and idx>0:
          print_progress(time.time()-start, idx, len(grid_data_loader))
      prob = torch.cat(l, 0)
   
    prob = prob.view(int(2 * limit / step), int(2 * limit / step))
    prob[prob!=prob]=0 #set nan probabilities to 0

    prob+=1e-20
    prob = prob/1024/1024
    nll = - gt_prob * np.log(prob.cpu().data.numpy())/np.log(2)
    nll = np.sum(nll)
    print('Negative Log Likelihood' , nll-20)

    prob /= torch.sum(prob)
    prob = prob.clamp(max=max_prob)

    prob = prob.cpu().data.numpy()

    fig = plt.figure(figsize=(18, 18))
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.imshow(gt_prob, extent=(-limit, limit, -limit, limit))

    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.imshow(prob, extent=(-limit, limit, -limit, limit))

    fig.tight_layout()


# %% [markdown] id="vkODCuaCzqBw"
# Let us first quickly evaluate our models using 1 importance sample.

# %% id="avKEPoK209lK"
plot_density2d(model8g, max_prob8g,prob8g, 1)


# %% id="6eW42o81b3g4"
plot_density2d(modelc, max_probc,probc, 1)


# %% id="s0Bzl5Z3b7wH"
plot_density2d(model2s, max_prob2s,prob2s, 1)

# %% [markdown] id="ki9ZZE65zSUX"
# When we use only 1 importance sample, then the result is essentially the negative ELBO. We will now better approximate the likelihood by taking 250 importance samples, which will take 250 times longer. This provides a more accurate evaluation of the model performance and gives us an idea of how tight of a bound the negative ELBO is.

# %% id="61CNugwGDDy7"
plot_density2d(model8g, max_prob8g,prob8g, 250) #should take around 20 minutes

# %% id="3n-fohH6fkNp"
plot_density2d(modelc, max_probc,probc, 250) #should take around 20 minutes

# %% id="-X2N6-G1fnjg"
plot_density2d(model2s, max_prob2s,prob2s, 250) #should take around 20 minutes


# %% [markdown] id="zDYelNNFxTJL"
# We can also qualitatively understand the unsupervised representation learning abilities of the VAE by visualizing the correspondence between the latent space and the color maps of the ground truth distribution. We also print out the probability mass of low-density ``dark pixels" to quantitatively evaluate how ``filled" the latent space is.
#

# %% id="5rKGQMRfydzb"
def visualize_latent(model, colormap_gt):
    limit=2
    step=1/256.0
    grid = torch.Tensor([[a, b] for a in np.arange(-limit, limit, step) for b in np.arange(-limit, limit, step)])
    grid_dataset = torch.utils.data.TensorDataset(grid.cuda())
    grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=20000, shuffle=False)
    colormap_gt = colormap_gt.reshape((1024*1024,3))
    l = []
    with torch.no_grad():
      for z ,in grid_data_loader:
        x,_ = model.decode(z) #find the value that each latent vector maps to
        l.append(x)
    x = torch.cat(l,0)
    x=(x*1024+512).long() #find the corresponding pixel in the color map 
    x[x<0]=0
    x[x>1023]=1023
    y = x[:,0]*1024+x[:,1]
    y = y.reshape((-1,1)).repeat((1,3))

    colormap_pred = torch.gather(torch.tensor(colormap_gt).cuda(), 0,y)
    pz = torch.exp(torch.sum(gaussian_log_prob(grid.cuda(), torch.zeros(grid.shape).cuda(), torch.zeros(grid.shape).cuda()),-1))
    pz/=torch.sum(pz)
    color = col.rgb_to_hsv(colormap_pred.cpu().numpy())[:,2]
    print('Dark Pixels', torch.sum(pz*(torch.tensor(color).cuda()<0.01).float()).item())


    colormap_pred = colormap_pred.reshape((1024,1024,3)).cpu().data.numpy()
    colormap_gt = colormap_gt.reshape((1024,1024,3))




    fig = plt.figure(figsize=(18, 18))
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.imshow(colormap_gt, extent=(-0.5, 0.5, -0.5, 0.5))

    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.imshow(colormap_pred, extent=(-limit, limit, -limit, limit))

    fig.tight_layout()



# %% id="B6I-CPRHFb0J"
visualize_latent(model8g,color8g)

# %% id="5ISzcTrcfAtE"
visualize_latent(modelc,colorc)

# %% id="WKqFQT4jfDEy"
visualize_latent(model2s,color2s)


# %% [markdown] id="tz7oJ5vejQeR"
# ## IWAEs
# We now switch out the negative ELBO for an approximation of the negative log likelihood using importance sampling.

# %% id="ITr9uPZAGdRT"
def train_iwae(dataset, model=None, epochs=60000, print_freq=1000):
  if model is None:
    model = VAE().cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=epochs/20,
                                                           min_lr=1e-8, verbose=True,
                                                           threshold_mode='abs')
  start=time.time()
  loss_ema=0
  for iteration in range(epochs): #train for 60k iterations
    data = torch.tensor(sample2d(dataset,200)).float().cuda()
    nll = model.importance_sampling(data,10)
    loss = torch.mean(nll)/math.log(2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_ema = 0.999*loss_ema + 0.001*loss
    #scheduler.step(loss_ema)
    if iteration == int(epochs*0.6)  or iteration == int(epochs*0.7) or iteration == int(epochs*0.8) or iteration == int(epochs*0.9):
       for param_group in optimizer.param_groups:
           param_group['lr'] /= 2
    if iteration%print_freq == 0:
      with torch.no_grad():
        print('Iteration %d. Loss: %f'%(iteration, loss))
        neg_elbo, l_rec, l_reg = model.compute_negative_elbo(data,0)
        print('Iteration %d. EMA: %f ELBO: %f L_rec: %f L_reg: %f'%(iteration, loss_ema, torch.mean(neg_elbo)/math.log(2),torch.mean(l_rec)/math.log(2), torch.mean(l_reg)/math.log(2)))
        print_progress(time.time()-start, iteration+1, epochs)

  return model


# %% id="6NbdjJV8GUKY"
model8g_iwae = train_iwae('8gaussians', None, 30000,400) #should take ~20 minutes to train

# %% id="mI--iqyMfydF"
modelc_iwae = train_iwae('checkerboard', None, 30000,400) #should take ~20 minutes to train

# %% id="uurxajhhJOsU"
model2s_iwae = train_iwae('2spirals', None, 30000,400) #should take ~20 minutes to train

# %% [markdown] id="cBrAwBPBjg0g"
# We see that compared to a Typical VAE, $D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}), p_\theta(\mathbf{z}))$ is lower, indicating that the variance of $q_\phi(\mathbf{z}|\mathbf{x})$ is higher than in a VAE. We now evaluate our model and check how accurate of an approximation our objective was to the true negative log likelihood.

# %% id="whYqBepvMZrt"
plot_density2d(model8g_iwae, max_prob8g,prob8g ,10)

# %% id="OwxTRZ25gFGG"
plot_density2d(modelc_iwae, max_probc,probc ,10)

# %% id="dRhQtzSogGPb"
plot_density2d(model2s_iwae, max_prob2s,prob2s ,10)

# %% id="sAtCDn3ThXnJ"
plot_density2d(model8g_iwae, max_prob8g,prob8g ,250) #should take ~20 minutes

# %% id="7ld8BN7ihZjD"
plot_density2d(modelc_iwae, max_probc,probc ,250) #should take ~20 minutes

# %% id="N5GonnEOhaB-"
plot_density2d(model2s_iwae, max_prob2s,prob2s ,250) #should take ~20 minutes

# %% [markdown] id="Tsi7H3uKjrP1"
# We now visualize the latent space of IWAE. We see that compared to Typical VAEs much less of the latent space is mapped to low-density inputs.

# %% id="KIozAKHUMflG"
visualize_latent(model8g_iwae,color8g)

# %% id="Hb3mpF6GgSP8"
visualize_latent(modelc_iwae,colorc)

# %% id="FYP4l--_gS0K"
visualize_latent(model2s_iwae,color2s)

# %% id="rRowqVPs9ONg"
print('Marginal KL',calc_marginal_kl(model8g_iwae, '8gaussians')) #should take ~1 minute

# %% id="KtwZzDvs9Q3A"
print('Marginal KL',calc_marginal_kl(modelc_iwae, 'checkerboard'))

# %% id="87tJNoMb9RH4"
print('Marginal KL',calc_marginal_kl(model2s_iwae, '2spirals'))
