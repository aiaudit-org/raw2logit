import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def mse(x,y):
    return ((x-y)**2).mean()

def gaussian_noise_entropies(t1, bins=20):
    all_MI= []
    all_mse = []
    for sigma in np.linspace(0,100,201):
        t2 = np.random.normal(t1.copy(), scale=sigma, size = t1.shape)
        hist_2d, x_edges, y_edges = np.histogram2d(
                                             t1.ravel(),
                                             t2.ravel(),
                                             bins=bins)
        all_mse.append(mse(t1,t2))
        MI = mutual_information(hist_2d)
        all_MI.append(MI)
        
    return np.array((all_MI)), np.array((all_mse))
        
def shifts_entropies(t1, bins=20):
    all_MI=[]
    all_mse=[]
    for N in np.linspace(1,50,50):
        N = int(N)
        temp_t2 = t1[:-N].copy()
        temp_t1 = t1[N:].copy()
        hist_2d, x_edges, y_edges = np.histogram2d(
                                     t1.ravel(),
                                     t2.ravel(),
                                     bins=bins)
        MI = mutual_information(hist_2d)
        
        all_mse.append(mse(temp_t1,temp_t2))
        all_MI.append(MI)
        
    return np.array((all_MI)), np.array((all_mse))

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def entropy(image, bins=20):
    image = image.ravel()   
    hist, bin_edges = np.histogram(image, bins = bins)
    hist = hist/hist.sum()
    entropy_term = np.where(hist != 0, hist*np.log(hist), 0)
    entropy = - np.sum(entropy_term)
    
    return entropy

# Gray Image
# t1 = np.array(Image.open("img.png"))[:,:,0].astype(float)

# Colour Image
t1 = np.array(Image.open("img.png").resize((255,255)))

perturb = "gauss"
show_image = True
bins=20

print(perturb)

# Identity
if perturb == "identity":
    t2 = t1
    title = "Identity"
    image1 = "Clean"
    image2 = "Clean"

# Poisson Noise on t2
if perturb == "poisson":
    t2 = np.random.poisson(t1)
    title = "Poisson Noise"
    image1 = "Clean"
    image2 = "Noisy"

# Gaussian Noise on t2
if perturb == "gauss":
    print(np.shape(t1))
    sigma = 50.0
    t2 = np.random.normal(t1.copy(), scale=sigma, size = t1.shape)
    if "grad" in locals():
        title = f"Gaussian Noise, grad= True, sigma = {sigma:.2f}"
    else:
        title = f"Gaussian Noise, sigma = {sigma:.2f}"
    image1 = "Clean"
    image2 = "Noisy"

if perturb == "box":
    sigma = 50.0
    mean = np.mean(t1)
    print(np.shape(t1))
    t2 = t1.copy()
    t2[30:220,50:120,:] = mean
    title = "Box with mean pixels"
    image1 = "Clean"
    image2 = "Noisy"


# Shift t2 on y axis
if perturb == "shift":
    N=30
    t2 = t1[:-N]
    t1 = t1[N:]
    title = "y shift"
    image1 = "Clean"
    image2 = "Shifted"

t2 = np.clip(t2,0,255).astype(int)

print("Correlation Coefficient: ",np.corrcoef(t1.ravel(), t2.ravel())[0, 1])

# 2D Histogram
hist_2d, x_edges, y_edges = np.histogram2d(
     t1.ravel(),
     t2.ravel(),
     bins=bins)

MI = mutual_information(hist_2d)

print("Mutual Information", MI)
print("Mean squared error:", mse(t1,t2))

if show_image == True:
    plt.figure()
    plt.imshow(np.hstack((t2, t1)))
    plt.title(title)
    
    plt.figure()
    
    plt.plot(t1.ravel(), t2.ravel(), '.')
    plt.xlabel(image1)
    plt.ylabel(image2)
    plt.title('I1 vs I2')
    
    plt.figure()
    plt.imshow((hist_2d.T)/hist_2d.max(), origin='lower')
    plt.xlabel(image1)
    plt.ylabel(image2)
    plt.xticks(ticks=np.linspace(0,bins-1,10), labels=np.linspace(x_edges.min(),x_edges.max(),10).astype(int))
    plt.yticks(ticks=np.linspace(0,bins-1,10), labels=np.linspace(y_edges.min(),y_edges.max(),10).astype(int))
    plt.title('p(x,y)')
    plt.colorbar()
    
    # Show log histogram, avoiding divide by 0
    plt.figure(figsize=(4,4))
    hist_2d_log = np.zeros(hist_2d.shape)
    non_zeros = hist_2d != 0
    hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
    plt.imshow((hist_2d_log.T)/hist_2d_log.max(), origin='lower')
    plt.xlabel(image1)
    plt.ylabel(image2)
    plt.xticks(ticks=np.linspace(0,bins-1,10), labels=np.linspace(x_edges.min(),x_edges.max(),10).astype(int))
    plt.yticks(ticks=np.linspace(0,bins-1,10), labels=np.linspace(y_edges.min(),y_edges.max(),10).astype(int))
    plt.title('log(p(x,y))')
    plt.colorbar()
    plt.show()

if perturb == "shift":
    mi_array, mse_array = shifts_entropies(t1, bins=bins)
    plt.figure()
    plt.plot(np.linspace(0,50,50), mi_array)
    plt.xlabel("y shift")
    plt.ylabel("Mutual Information")
    plt.figure()
    plt.plot(np.linspace(0,50,50), mse_array)
    plt.xlabel("y shift")
    plt.ylabel("Mean Squared Error")
    plt.show()
        
if perturb == "gauss":
    mi_array, mse_array = gaussian_noise_entropies(t1, bins= bins)
    plt.figure()
    plt.plot(np.linspace(0,100,201), mi_array)
    plt.xlabel("sigma")
    plt.ylabel("Mutual Information")
    plt.figure()
    plt.plot(np.linspace(0,100,201), mse_array)
    plt.xlabel("sigma")
    plt.ylabel("Mean Squared Error")
    plt.show()