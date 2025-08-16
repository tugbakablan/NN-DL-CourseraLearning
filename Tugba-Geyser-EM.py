#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


# Students names and ID: Tuğba KABLAN 5604269
# Students group: 12

# # A. EM and the Old Faithful geyser
# 
# ![Old_Faithful.jpg](attachment:Old_Faithful.jpg)
# 
# The duration of the eruptions varies in time as well as the delay between two consecutive eruptions. Our hypothesis is that these flares actually follow two distinct distributions that we will try to identify. We will thus use unsupervised learning of a variable in order to detect to which distribution each flare belongs.
# 

# In[2]:


# Loading dataset:

data = pkl.load( open('faithful.pkl', 'rb'))
X = data["X"] 

print("Size of the data", X.shape)
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.xlabel("Duration of the eruption")
plt.ylabel("Waiting time between eruptions")
# We can easily identify the two distributions, this is schoolbook example
# ... The goal will be to do so automatically


# ## A.1 Modeling
# 
# Following the previous visualisation, we choose to model the data using two latent distributions which are normal/gaussian distributions. We will have to determine their parameters.
# The likelihood of a normal distribution of dimension $N$ for an observation $\mathbf x \in \mathbb R^{N}$ is the following:
# 
# $$p(\mathbf x | \mu, \Sigma) = \frac{1}{(2 \pi)^{N / 2}|\Sigma|^{1 / 2}} e^{-\frac{1}{2}(\mathbf x-\mu) \Sigma^{-1}(\mathbf x-\mu)^{\top}}, \qquad  \mu \in \mathbb R ^N, \Sigma \in \mathbb R ^{N\times N}$$
# 
# The parameters of the normal distribution are:
# - $\mu \in \mathbb{R}^N$ the mean vector 
# - $\Sigma \in \mathbb{R}^{N\times N}$ the covariance matrix.
# 
# $|\Sigma|$ is the determinant of the matrix $\Sigma$.
# 
# ### Checking dimensions of the variables
# 
# Check rapidly on a draft paper that $(x-\mu) \Sigma^{-1}(\mathbf x-\mu)^{\top}$ is indeed a scalar value in $\mathbb{R}$. It is a good way to avoid implementation errors later on.
# 
# ### Writing the likelihood function
# 
# Write the code for `normal_2d`:`array(float) x array(float) x array(float) -> float` which takes as arguments `x, mu, Sig` and that returns the likelihood.
# 
# Note: You will look up the numpy functions that compute the determinant and the inverse of a matrice
# 

# In[8]:


import numpy as np

def normal_2d(x, mu, Sig):
    N = x.shape[0]
    diff = x - mu
    det_Sig = np.linalg.det(Sig)
    inv_Sig = np.linalg.inv(Sig)

    exponent = -0.5 * np.dot(diff.T, np.dot(inv_Sig, diff))
    coeff = 1 / (np.sqrt((2 * np.pi) ** N * det_Sig))

    return coeff * np.exp(exponent)


# In[9]:


mu  = np.array([1.,2])
Sig = np.array([[3., 0.],[0., 3.]])

x = np.array([1.,2])
print(normal_2d(x, mu, Sig)) # 0.053051647697298435
x = np.array([0,0])
print(normal_2d(x, mu, Sig)) # 0.023056151047594564


# ### Validating former results
# 
# Beyond the numbers obtained, does the order of the two probabilities seem valid? why?
# 
# ### Visualizing the contour lines of the gaussian
# 
# Visualizing contour lines on a 2D continuous distribution implies that we work in 3D. The following function gives the code to perform this operation (you do not have to add any code here).

# In[5]:


from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def plot_norm_2D(mu, Sig, bounds_min = np.array([-5, -5]), bounds_max = np.array([5, 5])):
    ngrid = 30
    x = np.linspace(bounds_min[0], bounds_max[0], ngrid)
    y = np.linspace(bounds_min[1], bounds_max[1], ngrid)
    X,Y = np.meshgrid(x,y)
    Z = np.array([normal_2d(np.array([x,y]), mu, Sig) 
                  for x,y in zip(X.flatten(), Y.flatten())]).reshape(ngrid, ngrid)
    fig = plt.gcf() # getting the current plot
    ax = fig.gca()
    ax.contour(X,Y,Z)



# In[10]:


# test on the distribution specified above
plt.figure()
plot_norm_2D(mu, Sig)


# In[11]:


# other test with two different variance values on the two axis
mu2  = np.array([0.,0.])
Sig2 = np.array([[1., 0.],[0., 4.]])
plt.figure()
plot_norm_2D(mu2, Sig2)


# In[12]:


# Last test with a positive covariance between the two dimension
# Note: be aware that the covariance matrix has to be symetric !!!
mu3  = np.array([0.,0.])
Sig3 = np.array([[1., 1.],[1., 4.]])
plt.figure()
plot_norm_2D(mu3, Sig3)


# ### Constructing by hand a gaussian that roughly fits our data
# 
# We only consider the upper right corner.
# 
# Estimating the mean: (4.25, 80) after inspecting the first figure.  
# 
# Estimating standard deviation $\sigma_1$ on first axis : ??? (Thumbrule : 2/3 of the data points should fits within $[\mu-\sigma_1; \mu+\sigma_1]$) 
# 
# Estimating standard deviation on second axis 2 : ??? (Thumbrule : 2/3 of the data points should fits within $[\mu-\sigma_2; \mu+\sigma_2]$) 
# 
# Estimating covariance $\rho$: ??? 
# 
# Once the covariance matrix has been estimated you should obtain:
# ![manual_gauss_2d.png](attachment:manual_gauss_2d.png)
# 
# **Note:** Be careful of not mixing up the standard deviation $\sigma$ and the variance $\sigma^2$.

# In[ ]:


Sig4 = np.array([[0.5**2, 0.],
                 [0., 10.**2]])
mu4  = np.array([4.25, 80.])
Sig4 = np.array([[0.25, 0.],   # 0.5^2
                 [0., 100.]])  # 10^2

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Duration of the eruption")
plt.ylabel("Waiting time between eruptions")
plot_norm_2D(mu4, Sig4, X.min(0), X.max(0))
# plt.savefig('manual_gauss_2d.png')


# ## A.2 Automated learning
# 
# Now let's stop testing by hand and try an automatic approach with the following properties:
# * it is robust to noisy data
# * it can be extended to multi-dimensional data that are difficult to display
# 
# We will use an EM approach and we will use the following notations.
# 
# Let $\Theta$ be the parameters of our two-class normal modeling:
# 
# $$\Theta = \{(\pi_0 \in \mathbb R, \mu_0 \in \mathbb{R}^2, \Sigma_0\in \mathbb{R}^{2\times 2}), (\pi_1, \mu_1, \Sigma_1)\}$$
# 
# where $\pi$ denotes the prior probabilities of the two classes.
# 
# For simplicity, we propose to store all $\pi$ in a single data structure, all $\mu$ in a single data structure and all $\Sigma$ in a structure so that we can easily increase or decrease the number of classes to be found in the future.

# ### EM : Step 0, initialisation
# 
# $\pi = [0.5, 0.5]$ is the most neutral and coherent initialization with respect to the data
# 
# For $\mu$, a classical strategy consists in starting from the mean of the cloud and adding $1$ on the dimensions for $\mu_1$ and subtracting $1$ on all dimensions for $\mu_2$.
# 
# For $\Sigma$, we propose to start from the variance matrix of the whole cloud for both models. We propose to initialize a 3D matrix in order to access the matrices of the models by doing `Sig[0]` and `Sig[1]`
# 
# The method `init` takes as argument `X` and returns `pi`, `mu` and `Sig`. 
# 
# **Note:** If you want to implement a more generic initialization, you have to take as additional argument the number of classes to predict (and possibly other things).

# In[14]:


def init(X):
    pi = np.array([0.5, 0.5])  # Equal prior probabilities

    mu_cloud = X.mean(axis=0)
    mu = np.array([
        mu_cloud + 1,   # First cluster shifted up
        mu_cloud - 1    # Second cluster shifted down
    ])

    cov = np.cov(X.T)  # Covariance of the full dataset
    Sig = np.array([cov, cov])  # One covariance matrix per class

    return pi, mu, Sig
# Check:
#[0.5 0.5] 
# [[ 4.48778309 71.89705882]
# [ 2.48778309 69.89705882]] 
# [[[  1.30272833  13.97780785]
#  [ 13.97780785 184.82331235]]
#
# [[  1.30272833  13.97780785]
#  [ 13.97780785 184.82331235]]]


# In[15]:


pi, mu, Sig = init(X)
print("pi:\n", pi)
print("\nmu:\n", mu)
print("\nSig:\n", Sig)


# ###  EM : the E step
# 
# We will now write the E step of the EM algorithm, which will allow us to estimate the parameters of the mixture of two-dimensional normal distribution. 
# 
# As seen in the course, let $Z$ be a random variable indicating which class/bidimensional normal law has generated the pair of data $\mathbf x \in \mathbb R^2$
# 
# Let $\Theta^t = \{\pi^t, \mu^t, \Sigma^t\}$ be the parameters at iteration $t$.
# 
# For each observation $\mathbf x$, we define $Q_i^{t+1}(0) = P(Z = 0|\mathbf x, \Theta^t)$, the probability of being from class $0$ for  an observation $\mathbf x$ given current parameters $\Theta^t$.
# 
# Applying bayes rule we can compute the probability for each class. 
#     $$Q_i^{t+1}(0) = \frac{p(\mathbf x | \mu_0, \Sigma_0) \pi_0}{\sum_{i=0}^1 p(\mathbf x | \mu_i, \Sigma_i) \pi_i},\qquad  Q_i^{t+1}(1) = \frac{p(\mathbf x | \mu_1, \Sigma_1) \pi_1}{\sum_{i=0}^1 p(\mathbf x | \mu_i, \Sigma_i) \pi_i}$$
#     
# Write the function `Q_i: np.array x np.array x np.array x np.array -> np.array` which takes as argument `X, pi, mu, Sig` and that returns the table of all $Q_i$ values.
# 
# Be carefull, `X` is the matrix containing all data points, `pi` is the vector of prior probabilities, `mu` and `Sig` are matrices contening the parameters for the two classes.  

# In[20]:


def Q_i(X, pi, mu, Sig):
    K = len(pi)  # number of classes
    N = X.shape[0]  # number of data points

    q = np.zeros((K, N))

    for k in range(K):
        for i in range(N):
            q[k, i] = pi[k] * normal_2d(X[i], mu[k], Sig[k])

    # Normalize over classes for each data point (i.e., columns)
    q /= q.sum(axis=0, keepdims=True)

    return q

# print(q[:,:5]) q for the 5 first data points
# [[0.02459605 0.03668168 0.05226123 0.01630238 0.49795917]
# [0.97540395 0.96331832 0.94773877 0.98369762 0.50204083]]


# ### EM : the M Step 
# #### (demonstration of the update formulas is given at the end)
# 
# Maximising the likelihood is a good opportunity to recall how the parameters for the bidimensional normal distribution are computed. We integrate here the reweighting from the $q$ values.
# 
# Thus: 
# $$ \mu_0 = \frac{\sum_i Q_i^{t+1}(0) \cdot \mathbf{x_i}}{\sum_i Q_i^{t+1}(0)}, \qquad \mu_1 = \frac{\sum_i Q_i^{t+1}(1) \cdot  \mathbf{x_i}}{\sum_i Q_i^{t+1}(1)} $$
# 
# $$ \Sigma_0 = \frac{\sum_i Q_i^{t+1}(0)\cdot  (\mathbf{x_i}-\mu_0)^T (\mathbf{x_i}-\mu_0) }{\sum_i Q_i^{t+1}(0)}, \qquad \Sigma_1 = \frac{\sum_i Q_i^{t+1}(1)\cdot  (\mathbf{x_i}-\mu_1)^T (\mathbf{x_i}-\mu_1)}{\sum_i Q_i^{t+1}(1)} $$
# 
# Take some time to draw the matrices shapes to verify the dimensions for $\Sigma$
# 
# 
# The prior probabilities correspond to the ratio of the probability masses $Q$:
# $$ \pi_0 = \frac{\sum_i Q_i^{t+1}(0) }{\sum_i Q_i^{t+1}(0) + Q_i^{t+1}(1) }, \qquad \pi_1 = \frac{\sum_i Q_i^{t+1}(1) }{\sum_i Q_i^{t+1}(0) +  Q_i^{t+1}(1)} $$
# 
# Write the function 
# `update_param: np.array x np.array x np.array x np.array x np.array -> np.array x np.array x np.array`
# that takes as argument `X`, `q`, `pi`, `mu`, `Sig` and return a new version of `pi`, `mu`, `Sig`

# In[22]:


def update_param(X, q, pi, mu, Sig):
    K = q.shape[0]  # number of classes
    N, D = X.shape  # number of data points and dimensions

    pi_new = np.zeros(K)
    mu_new = np.zeros((K, D))
    Sig_new = np.zeros((K, D, D))

    for k in range(K):
        q_k = q[k]  # shape (N,)
        sum_q_k = np.sum(q_k)

        # Update pi
        pi_new[k] = sum_q_k / N

        # Update mu
        mu_k = np.sum(q_k[:, np.newaxis] * X, axis=0) / sum_q_k
        mu_new[k] = mu_k

        # Update Sigma
        diff = X - mu_k  # shape (N, D)
        Sig_k = np.zeros((D, D))
        for i in range(N):
            Sig_k += q_k[i] * np.outer(diff[i], diff[i])
        Sig_k /= sum_q_k
        Sig_new[k] = Sig_k

    return pi_new, mu_new, Sig_new

# check:
# [0.51132321 0.48867679] 
# [[ 3.88361418 71.3886521 ]
# [ 3.07360826 70.38268397]] 
# [[[  1.04337668  12.40444673]
#  [ 12.40444673 162.96851264]]
#
# [[  1.22881404  15.10227603]
#  [ 15.10227603 205.78298546]]]


# In[24]:


q = Q_i(X, pi, mu, Sig)
pi_u, mu_u, Sig_u = update_param(X, q, pi, mu, Sig)

print(pi_u, "\n", mu_u, "\n", Sig_u)


# ### EM Algorithm: the loop
# 
# Alternate iterations of the E and the M step until convergence. Convergence is assessed by verifying that the updated parameters at $t+1$ are very close to the previous ones. We can do the convergence test on `mu` which is quite stable by checking the sum of the absolute differences: $| \mu^{t+1}_1 - \mu^t_1 | + |\mu^{t+1}_2 - \mu^t_2|$. Alternatively, one can also use the relative increase in the likelihood of the data to assess convergence.
# 
# The `EM` method takes `X` as argument and returns `pi`, `mu` and `Sig`. `nIterMax` is a default parameter to avoid infinite loop. Generally, iterative EM algorithms are coded with for loops rather than while loops for reliability.
# 
# The `saveParam` parameter will be useful in the following questions: you do not need to worry about it now.

# In[26]:


import os

def EM(X, nIterMax=100, saveParam=None):
    eps = 1e-3  # yakınsama eşiği
    pi, mu, Sig = init(X)

    for t in range(nIterMax):
        # E adımı
        q = Q_i(X, pi, mu, Sig)

        # M adımı
        pi_new, mu_new, Sig_new = update_param(X, q, pi, mu, Sig)

        # Yakınsama testi (mu vektörlerinin farkı)
        if np.sum(np.abs(mu_new - mu)) < eps:
            break

        pi, mu, Sig = pi_new, mu_new, Sig_new

        # opsiyonel olarak parametreleri kaydet
        if saveParam is not None:
            saveParam.append((pi.copy(), mu.copy(), Sig.copy()))

    return pi, mu, Sig


# Should converge in 22 iterations with a threshold at 1e-3 on absolute difference of the means
# [0.64411944 0.35588056] 
# [[ 4.28967856 79.96831569]
# [ 2.0364072  54.47870502]] 
# [[[ 0.16994739  0.94034165]
#  [ 0.94034165 36.04319866]]
#
# [[ 0.06918256  0.43532305]
#  [ 0.43532305 33.69834324]]]


# In[27]:


pi, mu, Sig = EM(X)
print(pi, "\n", mu, "\n", Sig)


# ### Displaying the results
# 
# A the end of the iterations you should get the following figure:
# 
# ![res_EM.png](attachment:res_EM.png)

# In[29]:


plt.figure()

plt.scatter(X[:,0], X[:,1])
plt.xlabel("Duration of the eruption")
plt.ylabel("waiting time between eruptions")
plot_norm_2D(mu[0], Sig[0], X.min(0), X.max(0) ) # plotting model 1
plot_norm_2D(mu[1], Sig[1], X.min(0), X.max(0) ) # plotting model 2
plt.savefig('res_EM.png')


# ## A-3 Small animated gif showing model convergence
# 
# The idea is to use the `saveParam` parameter of the EM method to save the parameters of the model at each iteration and to be able to trace all the figures corresponding to all the steps of the algorithm.
# 
# * you have to save the figures in a separate directory, otherwise your directory  will be a mess
# * you have to give explicit file names to be able to retrieve figures in the right order
# * you have to use pickle, because serialization is fantastic in this kind of case.
# 
# 
# The idea is to call: `pi, mu, Sig = EM(X, saveParam="params/faithful")`
# 
# You will have to create the subdirectory `params` and then save the parameters in:
# `params/faithful1.pkl`, `params/faithful2.pkl`, ...
#  
# In order to respect all the constraints, we suggest you to insert the following code in the EM method:
# ```python
# import os # Adding the library before starting the method 
# 
#      # detecting saving param
#      if saveParam != None:     
#           # creating subdirectory
#           if not os.path.exists(saveParam[:saveParam.rfind('/')]):
#                  os.makedirs(saveParam[:saveParam.rfind('/')])
#           pkl.dump({'pi':pi_u, 'mu':mu_u, 'Sig': Sig_u},\
#                     open(saveParam+str(i)+".pkl",'wb'))     # serializing
# ```
# 
# **Note:** all the code is given, you have to check that you are saving the right variables.

# 
# 

# In[44]:


import os
import pickle as pkl

def EM(X, nIterMax=100, saveParam=None):
    eps = 1e-3  # yakınsama eşiği
    pi, mu, Sig = init(X)

    # Eğer saveParam verilmişse, ilgili klasörü oluşturuyoruz
    if saveParam is not None:
        if not os.path.exists(saveParam[:saveParam.rfind('/')]):
            os.makedirs(saveParam[:saveParam.rfind('/')])

    for t in range(nIterMax):
        # E adımı
        q = Q_i(X, pi, mu, Sig)

        # M adımı
        pi_new, mu_new, Sig_new = update_param(X, q, pi, mu, Sig)

        # Yakınsama testi (mu vektörlerinin farkı)
        if np.sum(np.abs(mu_new - mu)) < eps:
            break

        pi, mu, Sig = pi_new, mu_new, Sig_new

        # Eğer saveParam verilmişse, her iterasyon sonunda parametreleri kaydediyoruz
        if saveParam is not None:
            # Parametreleri pickle ile kaydediyoruz
            pkl.dump({'pi': pi, 'mu': mu, 'Sig': Sig}, open(f"{saveParam}{t+1}.pkl", 'wb'))

    return pi, mu, Sig


# In[ ]:


import os
print(os.path.exists(r"C:/Users/TUĞBA KABLAN/Desktop/Machine Learning/ML- Geyser Project/faithful1.pkl"))


# In[58]:


import os
import pickle as pkl
import matplotlib.pyplot as plt
import imageio
import numpy as np
from matplotlib.patches import Ellipse

# İlk pkl dosyasından X verisini al
first_pkl = r"C:/Users/TUĞBA KABLAN/Desktop/Machine Learning/ML- Geyser Project/faithful.pkl"
if not os.path.exists(first_pkl):
    raise FileNotFoundError(f"İlk dosya bulunamadı: {first_pkl}")

with open(first_pkl, 'rb') as file:
    params = pkl.load(file)
    if 'X' in params:
        X = params['X']
    else:
        raise ValueError("X verisi .pkl dosyasında bulunamadı.")

# 2D Gaussian fonksiyonunu çizdirme
def plot_norm_2D(mu, Sigma, Xmin, Xmax, ax=None):
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    for n_std in range(1, 4):
        width, height = 2 * n_std * np.sqrt(vals)
        ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellip)

def create_gif(param_dir, output_gif='convergence.gif'):
    images = []

    # Parametre dosyalarını kontrol et
    for i in range(1, 101):
        param_file = os.path.join(param_dir, f"faithful{i}.pkl")
        if not os.path.exists(param_file):
            print(f"Dosya yok: {param_file}")
            break  # Dosya yoksa döngüyü sonlandır

        with open(param_file, 'rb') as file:
            try:
                params = pkl.load(file)
                pi = params['pi']
                mu = params['mu']
                Sig = params['Sig']
            except KeyError as e:
                print(f"Parametre hatası: {e} - Dosya: {param_file}")
                continue  # Parametre hatası varsa bir sonraki dosyaya geç

        # Görselleştirme
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
        plot_norm_2D(mu[0], Sig[0], X.min(0), X.max(0), ax)
        plot_norm_2D(mu[1], Sig[1], X.min(0), X.max(0), ax)

        ax.set_title(f'Iteration {i}')
        ax.set_xlabel("Duration of the eruption")
        ax.set_ylabel("Waiting time between eruptions")

        img_path = f"temp_{i}.png"
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

        images.append(imageio.imread(img_path))
        os.remove(img_path)

    if images:
        imageio.mimsave(output_gif, images, duration=0.5)
        print(f"GIF başarıyla oluşturuldu: {output_gif}")
    else:
        print("Hiçbir görsel oluşturulamadı.")

# GIF'i oluştur
param_dir = r"C:/Users/TUĞBA KABLAN/Desktop/Machine Learning/ML- Geyser Project"
output_gif = r"C:/Users/TUĞBA KABLAN/Desktop/Machine Learning/ML- Geyser Project/convergence.gif"
create_gif(param_dir, output_gif)


# In[59]:


pi, mu, Sig = EM(X, saveParam="params/faithful")


# You can then create the method to load parameters and create the animation using the code snippet below.
# 
# That does not always work out straight of the box... If it works you should obtain:
# ![old_faithful.gif](attachment:old_faithful.gif)

# In[63]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import glob
import numpy as np

# 2D Gaussian fonksiyonu
def plot_norm_2D(mu, Sigma, Xmin, Xmax, ax=None):
    if ax is None:
        ax = plt.gca()

    # Eigenvalues and eigenvectors of the covariance matrix
    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Plot 1, 2, and 3 standard deviations
    for n_std in range(1, 4):
        width, height = 2 * n_std * np.sqrt(vals)
        ellip = plt.matplotlib.patches.Ellipse(
            xy=mu, width=width, height=height, angle=theta, edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellip)

# Animasyon fonksiyonu
def create_animation(X, params_path, fname):
    nbiter = len(glob.glob(params_path + "/*.pkl"))  # .pkl dosyalarının sayısını alıyoruz
    print(nbiter)
    fig = plt.figure()

    # Grafiğin limitlerini ayarlıyoruz
    plt.xlim(X[:, 0].min() - (X[:, 0].mean() * 0.05), X[:, 0].max() + (X[:, 0].mean() * 0.05))
    plt.ylim(X[:, 1].min() - (X[:, 1].mean() * 0.05), X[:, 1].max() + (X[:, 1].mean() * 0.05))
    plt.xlabel("Duration of the eruption")
    plt.ylabel("Waiting time between eruptions")

    # Animasyon için fonksiyon
    def animate(i):
        print(f"Animating step: {i}")
        # .pkl dosyasını yükle
        param_file = f"{params_path}/{fname}{i}.pkl"
        if not os.path.exists(param_file):
            print(f"Dosya bulunamadı: {param_file}")
            return

        with open(param_file, 'rb') as file:
            data = pkl.load(file)

        # Grafiği temizle
        plt.clf()

        # Yeni bir eksen (axis) oluştur
        ax = fig.gca()
        ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
        ax.text(X[:, 0].max() * 0.75, X[:, 1].min() * 1.25, f'step = {i}')

        # Gaussianları çiz
        for j in range(len(data['mu'])):
            plot_norm_2D(data['mu'][j], data['Sig'][j], X.min(0), X.max(0), ax)

        return ax

    # FuncAnimation kullanarak animasyonu oluştur
    ani = animation.FuncAnimation(fig, animate, frames=nbiter, interval=300, repeat=True)

    # Animasyonu kaydet
    ani.save(params_path + '/animation.gif', bitrate=4000)
    print(f"Animasyon başarıyla oluşturuldu ve kaydedildi: {params_path}/animation.gif")

# X verisini yükle
X = np.random.rand(100, 2)  # Bu sadece örnek veri, kendi verinizi buraya yükleyin

# GIF'i oluştur
create_animation(X, "C:/Users/TUĞBA KABLAN/Desktop/Machine Learning/ML- Geyser Project/params", "faithful")


# ## A-4 Influence of initialisation parameters
# 
# EM is very often associated with non-convex likelihood functions. This means that there are multiples maxima and that a bad initialization can have disastrous consequences on the convergence of the algorithm.
# 
# Let's imagine a new initialization function where the initial averages are initialized in an unfavorable way, i.e. orthogonally to the natural separability of the data.
# 
# The data are really easy to separate and the code will continue to work... But we notice that convergence is much slower.
# 
# ![old_faithful_bad.gif](attachment:old_faithful_bad.gif)

# In[ ]:


import numpy as np
from sklearn.cluster import KMeans

def kmeans_init(X, k=2):
    # K-means initialization
    kmeans = KMeans(n_clusters=k).fit(X)
    pi = np.array([0.5, 0.5])  # Assuming equal priors initially

    # Means (mu) are initialized to K-means centroids
    mu = kmeans.cluster_centers_

    # Covariances (Sigma) are initialized to the covariance of the data
    Sig = np.zeros((2, 2, 2))  # Two clusters, each with a 2x2 covariance matrix
    Sig[0, :, :] = np.cov(X.T)
    Sig[1, :, :] = np.cov(X.T)

    return pi, mu, Sig


# # B- Working with 4 classes
# 
# We can construct a new toy dataset where there are 4 classes to detect.  
# 
# Change your code to take this case into account.
# 
# ![data_gauss.png](attachment:data_gauss.png)

# In[67]:


def generating_4_gaussians():
    n = 50 # nb points for each normal distribution

    Sig1 = np.array([[1, 0],[0, 1]])
    mu1  = np.array([-3, -3])
    Sig2 = np.array([[1.5, 0.5],[0.5, 1.5]])
    mu2  = np.array([3, -3])
    Sig3 = np.array([[1.2, -0.5],[-0.5, 1.2]])
    mu3  = np.array([-3, 3])
    Sig4 = np.array([[1.2, 0],[0, 1.2]])
    mu4  = np.array([3, 3])

    X = np.vstack((np.random.randn(n,2)@Sig1 +mu1, np.random.randn(n,2)@Sig2 +mu2,\
                   np.random.randn(n,2)@Sig3 +mu3, np.random.randn(n,2)@Sig4 +mu4))
    return X

Xg = generating_4_gaussians()

plt.figure()
plt.scatter(Xg[:,0],Xg[:,1])


# In[79]:


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import glob

# 4 Gaussian dağılımından veri seti oluştur
def generating_4_gaussians():
    n = 50  # Her normal dağılım için nokta sayısı
    Sig1 = np.array([[1, 0], [0, 1]])  # 1. Gaussian için kovaryans
    mu1  = np.array([-3, -3])  # 1. Gaussian için ortalama
    Sig2 = np.array([[1.5, 0.5], [0.5, 1.5]])  # 2. Gaussian için kovaryans
    mu2  = np.array([3, -3])  # 2. Gaussian için ortalama
    Sig3 = np.array([[1.2, -0.5], [-0.5, 1.2]])  # 3. Gaussian için kovaryans
    mu3  = np.array([-3, 3])  # 3. Gaussian için ortalama
    Sig4 = np.array([[1.2, 0], [0, 1.2]])  # 4. Gaussian için kovaryans
    mu4  = np.array([3, 3])  # 4. Gaussian için ortalama

    X = np.vstack((
        np.random.randn(n, 2) @ Sig1 + mu1,
        np.random.randn(n, 2) @ Sig2 + mu2,
        np.random.randn(n, 2) @ Sig3 + mu3,
        np.random.randn(n, 2) @ Sig4 + mu4
    ))
    return X

Xg = generating_4_gaussians()

# Veriyi görselleştir
plt.figure()
plt.scatter(Xg[:, 0], Xg[:, 1], alpha=0.6, s=20)
plt.title("4 Gaussian Dağılımı ile Toy Veri Seti")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# 4 sınıf için inisyalizasyon fonksiyonu
def init_4_classes(X):
    pi = np.array([0.25, 0.25, 0.25, 0.25])  # 4 sınıf için eşit priors
    mu1 = X.mean(0) + [0.1, -5]
    mu2 = X.mean(0) + [-0.1, 5]
    mu3 = X.mean(0) + [5, 0]
    mu4 = X.mean(0) + [-5, 0]
    mu = np.vstack((mu1, mu2, mu3, mu4))  # Ortalamaları birleştir
    Sig = np.zeros((4, 2, 2))
    Sig[0, :, :] = np.cov(X.T)
    Sig[1, :, :] = np.cov(X.T)
    Sig[2, :, :] = np.cov(X.T)
    Sig[3, :, :] = np.cov(X.T)
    return pi, mu, Sig

# EM algoritmasını çalıştır
pi, mu, Sig = init_4_classes(Xg)
pi, mu, Sig = EM(Xg, saveParam="faithful")

# Animasyonu oluştur
create_animation(Xg, "params_4_classes", "faithful")



# If your functions are robust, you should just have to change initialisation:
# 
# * pi = 4 values of same probability
# * mu = 4 vectors of dimension 2 shifted with respect to the global mean by [1,1], [-1,1], [1,-1], [-1,-1]
# * Sig = Covariance matrix of all data points, same for all classes
# 
# ![gauss.gif](attachment:gauss.gif)

# In[81]:


import numpy as np

def init(X):
    # pi: Four equal probabilities for each class
    pi = np.array([0.25, 0.25, 0.25, 0.25])  # 4 classes, equal probability

    # mu: Four means, shifted relative to the global mean by the specified values
    global_mean = X.mean(0)
    mu1 = global_mean + [1, 1]
    mu2 = global_mean + [-1, 1]
    mu3 = global_mean + [1, -1]
    mu4 = global_mean + [-1, -1]
    mu = np.vstack((mu1, mu2, mu3, mu4))  # Stack the means into a matrix

    # Sig: Covariance matrix of all data points, same for all classes
    Sig = np.cov(X.T)  # Covariance matrix of the data
    Sig = np.array([Sig] * 4)  # Create 4 identical covariance matrices for each class

    return pi, mu, Sig
# Assuming the EM function is defined elsewhere and works as expected.
pi, mu, Sig = EM(Xg, saveParam="params_gauss/gauss")

# Create the animation
create_animation(Xg, "params_gauss", "gauss")


# ### Trying bad initialisation parameters
# 
# * mu = 4 vectors of dimension 2 shifted with respect to the global mean by [4,2], [3,4], [0,0], [-5,0]
# 
# 
# Starting from the wrong points does not necessarily leads to the correct estimates
# ![gauss_bad.gif](attachment:gauss_bad.gif)
# 

# In[82]:


import numpy as np

def init(X):
    # pi: Four equal probabilities for each class (same as before)
    pi = np.array([0.25, 0.25, 0.25, 0.25])  # 4 classes, equal probability

    # mu: Bad initialization for the means with the specified shifts
    global_mean = X.mean(0)
    mu1 = global_mean + [4, 2]
    mu2 = global_mean + [3, 4]
    mu3 = global_mean + [0, 0]
    mu4 = global_mean + [-5, 0]
    mu = np.vstack((mu1, mu2, mu3, mu4))  # Stack the means into a matrix

    # Sig: Covariance matrix of all data points, same for all classes
    Sig = np.cov(X.T)  # Covariance matrix of the data
    Sig = np.array([Sig] * 4)  # Create 4 identical covariance matrices for each class

    return pi, mu, Sig
# Running the EM algorithm with bad initialization
pi, mu, Sig = EM(Xg, saveParam="params_gauss_bad/gauss")

# Creating the animation with the saved parameters
create_animation(Xg, "params_gauss_bad", "gauss")


# # C- Demonstration of the formulas for the M Step
# 
# **Note that the two-dimensional normal formula has been flattened: the two dimensions of the vectors are expressed as $(x,z)$, the means are $\mu_{x},\mu_{z}$ and the variance matrix $\Sigma$ is replaced by the standard deviations on both dimensions and the linear correlation coefficient: $\Sigma = \begin{pmatrix} \sigma_x & \rho \\ \rho & \sigma_z \end{pmatrix}$**.
# 
# We assume here that the pairs of duration and waiting time $(x,z)$ follow a mixture of
#  two-dimensional normal distributions. The probability distribution
# probability distribution that we want to estimate is : 
# 
# $$P(x,z|\Theta) = \pi_0 
# \mathcal{N}(\mu_{x0},\mu_{z0},\sigma_{x0},\sigma_{z0},\rho_0) (x,z) + \pi_1
# \mathcal{N}(\mu_{x1},\mu_{z1},\sigma_{x1},\sigma_{z1},\rho_1) (x,z)$$
# 
# We will note $f(\mu_{x},\mu_{z},\sigma_{x},\sigma_{z},\rho)$ the function
# of a two-dimensional normal distribution:
# 
# $$ \begin{array}{l}
#   f_{\mu_x,\mu_z,\sigma_x,\sigma_z,\rho}(x,z) = 
#   \frac{1}{2 \pi \sigma_x \sigma_z \sqrt{1-\rho^2}} \\
#   \quad\quad\quad\quad
#   \exp \left\{ -\frac{1}{2(1-\rho^2)}
#     \left[ \left(\frac{x-\mu_x}{\sigma_x} \right)^2
#       - 2\rho\frac {(x-\mu_x)(z-\mu_z)}{\sigma_x \sigma_z}
#       + \left( \frac {z-\mu_z}{\sigma_z} \right)^2
#     \right]
#   \right\}.
# \end{array}$$
# 
# Therefore, the logarithm of this function is:
# 
# $$\begin{array}{@{}l@{}}
#   \log f_{\mu_x,\mu_z,\sigma_x,\sigma_z,\rho}(x,z) = 
#   -\log (2\pi) -\log(\sigma_x) -\log(\sigma_z) 
#   -\frac{1}{2} \log(1-\rho^2) \\
#   \quad\quad\quad\quad
#   -\frac{1}{2(1-\rho^2)}
#   \left[ \left(\frac{x-\mu_x}{\sigma_x} \right)^2
#     - 2\rho\frac {(x-\mu_x)(z-\mu_z)}{\sigma_x \sigma_z}
#     + \left( \frac {z-\mu_z}{\sigma_z} \right)^2
#   \right].
# \end{array}$$
# 
# The M step will compute:
# 
# $\displaystyle\mbox{Argmax}_{\Theta} \log(L^{t+1}(\mathbf{x}^o,\Theta)) =
# \mbox{Argmax}_{\Theta} \sum_{i=1}^n \sum_{k=0}^1 Q_i^{t+1}(y_k) \log 
# \left(\frac{p(x_i,z_i,y_k | \Theta)}{Q_i^{t+1}(y_k)}\right)$
# 
# $= \mbox{Argmax}_{\Theta} \sum_{i=1}^n 
# Q_i^{t+1}(y_0) \log ( \pi_0 
# f_{\mu_{x0},\mu_{z0},\sigma_{x0},\sigma_{z0},\rho_0} (x,z) ) +
# Q_i^{t+1}(y_1) \log ( \pi_1
# f_{\mu_{x1},\mu_{z1},\sigma_{x1},\sigma_{z1},\rho_1} (x,z) )$
# 
# Given that $\pi_1 = 1 - \pi_0$
# 
# $\displaystyle\frac{\partial\log(L^{t+1}(\mathbf{x}^o,\Theta))}{\partial \pi_0} 
# = \sum_{i=1}^n Q_i^{t+1}(y_0) \frac{1}{\pi_0} - 
# \sum_{i=1}^n Q_i^{t+1}(y_1) \frac{1}{1 - \pi_0} = 0$
# 
# Thus:
# 
# $$\pi_0 = \frac{\sum_{i=1}^n Q_i^{t+1}(y_0)}{\sum_{i=1}^n Q_i^{t+1}(y_0) +
#   Q_i^{t+1}(y_1)}\quad\quad\quad\quad(1)$$
# 
# and
# 
# $$\pi_1 = 1 - \pi_0 = 
# \frac{\sum_{i=1}^n Q_i^{t+1}(y_1)}{\sum_{i=1}^n Q_i^{t+1}(y_0) +
#   Q_i^{t+1}(y_1)}\quad\quad\quad\quad(2)$$
# 
# Now let's calculate how to update $\mu$:
# 
# $\displaystyle\frac{\partial\log(L^{t+1}(\mathbf{x}^o,\Theta))}{\partial\mu_0}=\frac{\partial}{\partial\mu_{x0}}\sum_{i=1}^n Q_i^{t+1}(y_0)\left(-\frac{1}{2(1-\rho_0^2)} \right)\left[\left(\frac{x_i-\mu_{x0}}{\sigma_{x0}}\right)^2-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]$
# 
# $\displaystyle\Longleftrightarrow\sum_{i=1}^n Q_i^{t+1}(y_0)\left(-\frac{1}{2(1-\rho_0^2)}\right)\left[2\left(\frac{x_i-\mu_{x0}}{\sigma_{x0}^2}\right) - 2\rho_0\frac {z_i-\mu_{z0}}{\sigma_{x0} \sigma_{z0}}\right] = 0$
# 
# which is equivalent to:
# 
# $$\sum_{i=1}^n Q_i^{t+1}(y_0) \left[ 2 \left(\frac{x_i - \mu_{x0}}{\sigma_{x0}} \right) - 2\rho_0\frac {z_i-\mu_{z0}}{\sigma_{z0}} \right] = 0 \quad\quad\quad\quad(3)$$
# 
# By symmetry, deriving with respect to {$\mu_{z0}$} we also have:
# 
# $$\sum_{i=1}^n Q_i^{t+1}(y_0) \left[ 2 \left(\frac{z_i - \mu_{z1}}{\sigma_{z1}} \right)
#   - 2\rho_0\frac {x_i-\mu_{x0}}{\sigma_{x0}} \right] = 0 \quad\quad\quad\quad(4)$$
# 
# If we add $\rho_0$ times the equation $(4)$ to the equation $(3)$,
# we get :
# 
# 
# $$\sum_{i=1}^n Q_i^{t+1}(y_0) \times 2 \left[ \frac{1 - \rho_0^2}{\sigma_{x0}}\right](\mu_{x0} - x_i) = 0$$
# 
# which is equivalent to:
# 
# $$\mu_{x0} = \frac{\sum_{i=1}^n Q_i^{t+1}(y_0) x_i}{\sum_{i=1}^n Q_i^{t+1}(y_0)}
# \quad\quad\quad\quad(5)$$
# 
# By symmetry we have:
# 
# $$\mu_{z0} = \frac{\sum_{i=1}^n Q_i^{t+1}(y_0) z_i}{\sum_{i=1}^n Q_i^{t+1}(y_0)}
# \quad\quad\quad\quad(6)$$
# 
# Let us now compute the expressions of $\sigma$ and $\rho$:
# 
# $\frac{\partial\log(L^{t+1}(\mathbf{x}^o,\Theta))}{\partial\sigma_{x0}}=\frac{\partial}{\partial\sigma_{x0}}\sum_{i=1}^nQ_i^{t+1}(y_0)\left\{-\log(\sigma_{x0})-\frac{1}{2(1-\rho_0^2)}\left[\left(\frac{x_i-\mu_{x0}}{\sigma_{x0}}\right)^2-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\}$
# 
# 
# $\displaystyle= \sum_{i=1}^n Q_i^{t+1}(y_0)\left\{-\frac{1}{\sigma_{x0}}-\frac{1}{2(1-\rho_0^2)}\left[-2\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^3}+2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}^2}\right]\right\}=0$
# 
# $\displaystyle\Longleftrightarrow \sum_{i=1}^n Q_i^{t+1}(y_0)\left\{-1-\frac{1}{2(1-\rho_0^2)}\left[-2\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}+2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\}=0$
# 
# which is equivalent to:
# 
# $$2(1-\rho_0^2)\sum_{i=1}^n Q_i^{t+1}(y_0)=\sum_{i=1}^n Q_i^{t+1}(y_0)\left\{\left[2\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\}\quad\quad\quad\quad(7)$$
# 
# By symmetry, when we derive with respect to $\sigma_{z0}$, we get:
# 
# $$2(1-\rho_0^2)\sum_{i=1}^n Q_i^{t+1}(y_0)=\sum_{i=1}^n Q_i^{t+1}(y_0)\left\{\left[2\frac{(z_i-\mu_{z0})^2}{\sigma_{z0}^2}-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\}\quad\quad\quad\quad(8)$$
# 
# By adding the 2 equations $(7)$ and $(8)$, we obtain:
# 
# $$4(1-\rho_0^2)\sum_{i=1}^nQ_i^{t+1}(y_0)=2\sum_{i=1}^nQ_i^{t+1}(y_0)\left\{\left[\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}+\frac{(z_i-\mu_{z0})^2}{\sigma_{z0}^2}\right]\right\}\quad\quad\quad\quad(9)$$
# 
# Finally, let us derive the log-likelihood with respect to $\rho_0$:
# 
# $\begin{array}{l}\displaystyle\frac{\partial\log(L^{t+1}(\mathbf{x}^o,\Theta))}{\partial\rho_{0}}=\frac{\partial}{\partial\rho_{x0}}\sum_{i=1}^n Q_i^{t+1}(y_0)\left\{-\frac{1}{2}\log(1-\rho_0^2)\right.\\\displaystyle\quad\quad\quad\quad-\frac{1}{2(1- \rho_0^2)}\left.\left[\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}+\frac{(z_i-\mu_{z0})^2}{\sigma_{z0}^2}\right]\right\}\end{array}$
# 
# $\begin{array}{l}\displaystyle=\sum_{i=1}^n Q_i^{t+1}(y_0)\left\{\frac{\rho_0}{1-\rho_0^2}\right.\\\displaystyle\quad\quad\quad-\frac{\rho_0}{(1- \rho_0^2)^2}\left[\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}+\frac{(z_i-\mu_{z0})^2}{\sigma_{z0}^2}\right]\\\displaystyle\quad\quad\quad+\frac{1}{1-\rho_0^2}\left.\left[\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\}=0\end{array}$
# 
# $\begin{array}{l}\displaystyle\Longleftrightarrow\sum_{i=1}^n Q_i^{t+1}(y_0)\left\{\rho_0\right.\\\displaystyle\quad\quad\quad-\frac{\rho_0}{(1- \rho_0^2)}\left[\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}-2\rho_0\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}+\frac{(z_i-\mu_{z0})^2}{\sigma_{z0}^2}\right]\\\displaystyle\quad\quad\quad+\left.\left[\frac{(x_i-\mu_{x0})(z_i-\mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\}=0\end{array}$
# 
# Replacing the 2nd term with the left-hand member of equation (9), we obtain :
# 
# $\displaystyle\sum_{i=1}^n Q_i^{t+1}(y_0)\left\{\rho_0 - 2 \rho_0+\left[\frac{(x_i - \mu_{x0})(z_i - \mu_{z0})}{\sigma_{x0}\sigma_{z0}}\right]\right\} = 0$
# 
# As a result,
# 
# $$\displaystyle\rho_0 = \frac{\sum_{i=1}^n Q_i^{t+1}(y_0)\frac{(x_i - \mu_{x0})(z_i - \mu_{z0})}{\sigma_{x0}\sigma_{z0}}}{\sum_{i=1}^n Q_i^{t+1}(y_0)}\quad\quad\quad\quad(10)$$
# 
# Let's write $\eta = \sum_{i=1}^n Q_i^{t+1}(y_0)(x_i - \mu_{x0})(z_i - \mu_{z0})$. Then:
# 
# $$\displaystyle\rho_0 = \frac{\eta}{\sigma_{x0}\sigma_{z0}\sum_{i=1}^n Q_i^{t+1}(y_0)}$$
# 
# Replacing $\rho_0$ by its value in equation (8), we obtain:
# 
# $\begin{array}{l}\displaystyle2\left(1-\left(\frac{1}{\sum_{i=1}^n Q_i^{t+1}(y_0)}\right)^2\frac{\eta^2}{\sigma_{x0}^2\sigma_{z0}^2}\right)\sum_{i=1}^n Q_i^{t+1}(y_0)=\\\displaystyle\quad\quad\quad\quad2 \left(\sum_{i=1}^n Q_i^{t+1}(y_0)\frac{(x_i-\mu_{x0})^2}{\sigma_{x0}^2}\right)-2\frac{\eta^2}{\sigma_{x0}^2\sigma_{z0}^2\sum_{i=1}^n Q_i^{t+1}(y_0)}\end{array}$
# 
# which is equivalent to:
# 
# $\displaystyle \sigma_{x0}^2 \sum_{i=1}^n Q_i^{t+1}(y_0) = \sum_{i=1}^n Q_i^{t+1}(y_0) (x_i-\mu_{x0})^2.$
# 
# Thus
# 
# $$\displaystyle \sigma_{x0}^2 = \frac{\sum_{i=1}^n Q_i^{t+1}(y_0)(x_i-\mu_{x0})^2}{\sum_{i=1}^n Q_i^{t+1}(y_0)}\quad\quad\quad\quad(11)$$
# 
# By symmetry
# 
# $$\displaystyle\sigma_{z0}^2 = \frac{\sum_{i=1}^n Q_i^{t+1}(y_0)(z_i-\mu_{z0})^2}{\sum_{i=1}^n Q_i^{t+1}(y_0)}\quad\quad\quad\quad(12)$$
# 
# 
# 

# In[88]:


import numpy as np
import matplotlib.pyplot as plt

# 4 sınıf için rastgele veri oluşturma
np.random.seed(42)

# Veri: x ve z koordinatları
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
X0 = np.random.multivariate_normal([4, 2], cov, 100)
X1 = np.random.multivariate_normal([3, 4], cov, 100)
X2 = np.random.multivariate_normal([0, 0], cov, 100)
X3 = np.random.multivariate_normal([-5, 0], cov, 100)

Xg = np.vstack([X0, X1, X2, X3])

# Görselleştirme
plt.scatter(Xg[:, 0], Xg[:, 1])
plt.title('Veri Kümesi (X)')
plt.xlabel('x')
plt.ylabel('z')
plt.show()

# Başlangıç parametreleri (mu, sigma, pi)
mu = np.array([[4, 2], [3, 4], [0, 0], [-5, 0]])  # 4 vektör (mean) başlangıcı
sigma = np.array([[[1, 0.5], [0.5, 1]], [[1, 0.5], [0.5, 1]], [[1, 0.5], [0.5, 1]], [[1, 0.5], [0.5, 1]]])  # Aynı kovaryans matrisi
pi = np.array([0.25, 0.25, 0.25, 0.25])  # 4 sınıf için başlangıç olasılıkları

from scipy.stats import multivariate_normal

def e_step(X, pi, mu, sigma):
    # Her bir sınıfın olasılıklarını hesaplayalım
    N = X.shape[0]
    K = len(pi)  # Sınıf sayısı
    gamma = np.zeros((N, K))  # Her bir örnek için sınıf olasılıkları

    for k in range(K):
        mvn = multivariate_normal(mean=mu[k], cov=sigma[k])
        gamma[:, k] = pi[k] * mvn.pdf(X)

    # Normalize et (her bir örneğin tüm sınıflara ait olasılıklarının toplamı 1 olmalı)
    gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]
    return gamma

def m_step(X, gamma):
    N, D = X.shape
    K = gamma.shape[1]  # Sınıf sayısı

    pi_new = gamma.sum(axis=0) / N  # Her bir sınıf için pi

    mu_new = np.dot(gamma.T, X) / gamma.sum(axis=0)[:, np.newaxis]  # Her sınıf için mean

    sigma_new = np.zeros((K, D, D))
    for k in range(K):
        X_centered = X - mu_new[k]
        sigma_new[k] = np.dot(gamma[:, k] * X_centered.T, X_centered) / gamma[:, k].sum()

    return pi_new, mu_new, sigma_new

def em_algorithm(X, pi, mu, sigma, max_iter=100):
    for _ in range(max_iter):
        # E Adımı
        gamma = e_step(X, pi, mu, sigma)

        # M Adımı
        pi, mu, sigma = m_step(X, gamma)

    return pi, mu, sigma

# EM algoritmasını çalıştırma
pi_new, mu_new, sigma_new = em_algorithm(Xg, pi, mu, sigma)

# Sonuçları yazdırma
print("Güncellenmiş pi:", pi_new)
print("Güncellenmiş mu:", mu_new)
print("Güncellenmiş sigma:", sigma_new)

# Öngörülen sınıf etiketlerini hesaplayalım
gamma_final = e_step(Xg, pi_new, mu_new, sigma_new)

# En yüksek olasılıkla sınıf etiketlerini seçme
labels = np.argmax(gamma_final, axis=1)

# Veri noktalarını ve tahmin edilen sınıfları görselleştirme
plt.scatter(Xg[:, 0], Xg[:, 1], c=labels, cmap='viridis')
plt.title('Veri Kümesi ve Tahmin Edilen Sınıflar')
plt.xlabel('x')
plt.ylabel('z')
plt.show()

