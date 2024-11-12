import numpy as np
import matplotlib.pyplot as plt
from bayes_coloc.continuous_donut_interaction import DonutInteraction
from bayes_coloc.plot_matching_lines import plot_pair_prob
from scipy.special import gammaln


%load_ext autoreload
%autoreload 2

np.random.seed(10)

d = 2

# set parameters
area = 1
scale = 0.02
radius = 0.02
gamma = 10
alpha = 0
beta = 0
true_params = {'gamma': gamma, 'alpha': alpha, 'beta': beta,
    'scale': scale, 'radius': radius, 'area': area}

### Data generation 

n_gamma = np.random.poisson(lam=true_params['gamma']*area)
n_mu = np.random.poisson(lam=true_params['alpha']*area)
n_nu = np.random.poisson(lam=true_params['beta']*area)

# Sample points in space 
hatmu = np.random.random(size=(n_mu, d)) 
hatnu = np.random.random(size=(n_nu, d)) 
hatgamma_x = np.random.random(size=(n_gamma, d))
random_angles = 2*np.pi*np.random.random(size=n_gamma)
random_directions = np.vstack((np.cos(random_angles), np.sin(random_angles))).T
random_shift = random_directions * radius
hatgamma_y = hatgamma_x + np.random.normal(scale=scale, size=(n_gamma, d)) + random_shift

# Merge marginals
x = np.vstack((hatgamma_x, hatmu))
y = np.vstack((hatgamma_y, hatnu))

nx, ny = len(x), len(y)

# now shuffle the data
perm_x = np.random.permutation(nx)
perm_y = np.random.permutation(ny)
x = x[perm_x]
y = y[perm_y]

# COMPUTE (TRUE) MATCHING
unpermuted_matching_mat = np.zeros((nx, ny))
unpermuted_matching_mat[range(n_gamma), range(n_gamma)] = 1
matching_mat = unpermuted_matching_mat[:, perm_y]
matching_mat = matching_mat[perm_x, :]


fig, axis = plt.subplots(ncols=3, figsize=(12, 4))
marker = "o"
alpha = 0.5
linewidth = 2
plan = np.eye(len(hatgamma_y))
plot_pair_prob(hatgamma_x, hatgamma_y, plan, l=linewidth, 
               ax=axis[0], scatter_alpha=alpha)
axis[0].set_aspect("equal")
h = 0.05

axis[0].set_title("Paired Poisson")
axis[1].set_title("Mixture")
axis[2].set_title("Observed Data")

for ax in axis:
    ax.set_aspect("equal")
    ax.set_xlim(-h, np.sqrt(area)+h)
    ax.set_ylim(-h, np.sqrt(area)+h)
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
plot_pair_prob(x, y, matching_mat, ax=axis[1], l=linewidth, scatter_alpha=alpha)
no_matching = np.zeros((nx, ny))
plot_pair_prob(x, y, no_matching, ax=axis[2], l=linewidth, scatter_alpha=alpha)
axis[1].set_title("Mixture")
axis[2].set_title("Observed Data")
fig.suptitle("Setup of the problem", fontsize=16)
# thight layout
plt.tight_layout()
# plt.savefig("setup_plot_v2.pdf")
plt.show()


# %%
# print relevant data
print("lambda gamma: ", lam_gamma, "lambda mu: ", lam_mu, "lambda nu: ", lam_nu)
print("Number of pairs: ", n_gamma, "Number of single A: ", n_mu, "Number of single B: ", n_nu)
print("Number of points in x: ", nx, "Number of points in y: ", ny)
print("Standard deviation", scale)
print("Radius: ", radius)

# %%
# define proposal, prior and starting parameters
scale_interval = [0., 0.1]
radius_interval = [0, 0.1]

area = 1
hyper_theta = len(x)/2

def log_prior(params):
    """ Defines a log prior for the parameters of the model (up to a constant)"""
    
    alpha = params["alpha"]
    gamma = params['gamma']
    beta = params["beta"]
    radius = params['radius']
    scale = params['scale']
    
    # hyperparameters
    k = 1
    theta = hyper_theta
    def ld_gamma(x, k, theta):  
        a = k
        b = 1/theta
        if x <= 0:
            return -np.inf
        else:
            return a*np.log(b) - gammaln(a) + (a-1)*np.log(x) - b*x

    # check if lam parameters are valid
    log_prob = 0
    
    log_prob += ld_gamma(alpha, k, theta)
    log_prob += ld_gamma(gamma, k, theta)
    log_prob += ld_gamma(beta, k, theta)
    
    if scale_interval[0] < scale < scale_interval[1]:
        log_prob += 0
    else:
        log_prob = -np.inf
    if radius_interval[0] < radius < radius_interval[1]:
        log_prob += 0
    else:
        log_prob = -np.inf
    return log_prob

def proposal(params):
    """ Defines a proposal for the parameters of the model """
    scale_temp = 0.0
    radius_temp = 0.00
    intensity_temp = 1
    
    scale = params['scale']
    radius = params['radius']
    alpha = params["alpha"]
    gamma = params['gamma']
    beta = params["beta"]
    
    # new proposal for the scale
    scale_new = scale + np.random.normal(scale=scale_temp)
    new_radius = radius + np.random.normal(scale=radius_temp)
    
    # new proposal for the intensities
    alpha_new = alpha + np.random.normal(scale=intensity_temp)
    beta_new = beta + np.random.normal(scale=intensity_temp)
    gamma_new = gamma + np.random.normal(scale=intensity_temp)
    
    return {"alpha": alpha_new, "beta": beta_new, "gamma": gamma_new, "scale": scale_new, "radius": new_radius, "area": area}

# start_params = {"alpha": hyper_theta, "beta": hyper_theta, "gamma": hyper_theta,
#                 "scale": 0.05, "radius": 0.05, "area": area}
start_params = true_params.copy()

swap_params = start_params.copy()



# %%
np.random.seed(0)

chain = DonutInteraction(x, y,start_params, swap_params, log_prior, proposal)

# %%
chain.run(n_samples=5000, burn_in=0)
print(len(chain.param_trajectory))

# %%
print("chain acceptance rate", chain.latent_acceptance_rate)
print("param acceptance rate", chain.param_acceptance_rate)

# %%
print(len(x), len(y), n_gamma)
print(n_gamma, n_mu, n_nu)

# %%
fig, axis = plt.subplots(ncols=2, figsize=(8, 4))
marker = "o"
alpha = 0.5
linewidth = 2
plan = np.eye(len(hatgamma_y))
plot_pair_prob(hatgamma_x, hatgamma_y, plan, l=linewidth, 
               ax=axis[0], scatter_alpha=alpha)
axis[0].set_aspect("equal")
h = 0.05



for ax in axis:
    ax.set_aspect("equal")
    ax.set_xlim(-h, 1+h)
    ax.set_ylim(-h, 1+h)
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
plot_pair_prob(x, y, matching_mat, ax=axis[0], l=linewidth, scatter_alpha=alpha)
no_matching = np.zeros((nx, ny))
plot_pair_prob(x, y, chain.pairing_probabilities(), ax=axis[1], l=linewidth, scatter_alpha=alpha)
axis[0].set_title("Ground Truth")
axis[1].set_title("Probability of pairing")
# thight layout
plt.tight_layout()
# plt.savefig("setup_plot_v2.pdf")
plt.show()


# %%
gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 0.5])
fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec, figsize=(10, 4))
n_max = np.minimum(nx, ny)

ax[1].set_visible(False)

pair_count = chain.pair_count_trajectory
ax[0].plot(np.arange(len(pair_count)), pair_count)
ax[0].set_ylim((0, n_max+2))
ax[0].hlines(y=n_gamma, xmin=0, xmax=len(pair_count), color="red")
bins = np.arange(n_max+2)-0.5
ax[2].hist(chain.pair_count_trajectory, bins=bins, alpha=0.5, label="MCMC", 
           orientation="horizontal", color="tab:blue", density=True)
xmax = np.max(np.histogram(pair_count, bins=bins, density=True)[0])
ax[2].hlines(y=n_gamma, xmin=0, xmax=xmax, color="red")
ax[2].set_ylim((0, n_max+2))
ax[2].set_yticklabels([])
plt.show()



# %%
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(chain.pairing_probabilities())
ax[1].imshow(matching_mat)

# %%
# scale parameter
gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 0.5])
fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec, figsize=(10, 4))
fig.suptitle("Scale parameter", fontsize=16)
ax[0].set_ylim(scale_interval)
ax[1].set_visible(False)

scale_trajectory = [params["scale"] for params in chain.param_trajectory]
ax[0].plot(np.arange(len(scale_trajectory)), scale_trajectory)
ax[0].hlines(y=true_params["scale"], xmin=0, xmax=len(scale_trajectory), color="red")

bins = np.linspace(scale_interval[0], scale_interval[1], 50)
ax[2].hist(scale_trajectory, bins=bins, alpha=0.5, label="MCMC", 
           color="tab:blue", density=True, orientation="horizontal")
xmax = np.max(np.histogram(scale_trajectory, bins=bins, density=True)[0])
ax[2].hlines(y=true_params["scale"], xmin=0, xmax=xmax, color="red")
ax[2].set_ylim(scale_interval)
ax[2].set_yticklabels([])
plt.show()



# %%
# do plot for radius parameter
gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 0.5])
fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec, figsize=(10, 4))
fig.suptitle("Radius parameter", fontsize=16)
# ax[0].set_ylim(radius_interval)
ax[1].set_visible(False)

radius_trajectory = [params["radius"] for params in chain.param_trajectory]
ax[0].plot(np.arange(len(radius_trajectory)), radius_trajectory)
ax[0].hlines(y=true_params["radius"], xmin=0, xmax=len(radius_trajectory), color="red")

bins = np.linspace(radius_interval[0], radius_interval[1], 50)
ax[2].hist(radius_trajectory, bins=bins, alpha=0.5, label="MCMC", 
           color="tab:blue", density=True, orientation="horizontal")
xmax = np.max(np.histogram(radius_trajectory, bins=bins, density=True)[0])
ax[2].hlines(y=true_params["radius"], xmin=0, xmax=xmax, color="red")
ax[2].set_ylim(radius_interval)
ax[2].set_yticklabels([])
plt.show()


# %%
# plot trajectory of radius and scale in 2D
fig, ax = plt.subplots(ncols=1, figsize=(10, 5))
ax.plot(radius_trajectory, scale_trajectory)
ax.scatter([true_params["radius"]], [true_params["scale"]], color="red")
ax.set_xlabel("Radius")
ax.set_ylabel("Scale")
ax.set_title("Trajectory in parameter space")



# %%
# Probability of A being paired with B
gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 0.5])
fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec, figsize=(10, 4))
fig.suptitle("Probability of pairing in the first channel", fontsize=16)
ax[0].set_ylim((0, 1))

ax[1].set_visible(False)
# Print the keys of the first element in chain.param_trajectory to verify the keys
print(chain.param_trajectory[0].keys())

# Probability of A being paired with B
p_a_paired_traj = [params["gamma"]/(params["gamma"]+params["alpha"]) for params in chain.param_trajectory]
p_a_paired_true = true_params["gamma"]/(true_params["gamma"] + true_params["alpha"])
ax[0].plot(np.arange(len(p_a_paired_traj)), p_a_paired_traj)
ax[0].hlines(y=p_a_paired_true, xmin=0, xmax=len(p_a_paired_traj), color="red")

bins = np.linspace(0, 1, 50)
ax[2].hist(p_a_paired_traj, bins=bins, alpha=0.5, label="MCMC", 
           color="tab:blue", density=True, orientation="horizontal")
xmax = np.max(np.histogram(p_a_paired_traj, bins=bins, density=True)[0])
ax[2].hlines(y=p_a_paired_true, xmin=0, xmax=xmax, color="red")
ax[2].set_ylim((0, 1))
ax[2].set_yticklabels([])
plt.show()

# %%
# probability of pairing in the second channel
gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 0.5])
fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec, figsize=(10, 4))
fig.suptitle("Probability of pairing in the second channel", fontsize=16)
ax[0].set_ylim((0, 1))

ax[1].set_visible(False)
p_b_pair_traj = [params["gamma"]/(params["gamma"]+params["alpha"]) for params in chain.param_trajectory]
p_b_pair_true = true_params["gamma"]/(true_params["gamma"] + true_params["alpha"])
ax[0].plot(np.arange(len(p_b_pair_traj)), p_b_pair_traj)
ax[0].hlines(y=p_b_pair_true, xmin=0, xmax=len(p_b_pair_traj), color="red")

bins = np.linspace(0, 1, 50)
ax[2].hist(p_b_pair_traj, bins=bins, alpha=0.5, label="MCMC", 
           color="tab:blue", density=True, orientation="horizontal")
xmax = np.max(np.histogram(p_b_pair_traj, bins=bins, density=True)[0])
ax[2].hlines(y=p_b_pair_true, xmin=0, xmax=xmax, color="red")
ax[2].set_ylim((0, 1))

ax[2].set_yticklabels([])
plt.show()



# %%


# %%




