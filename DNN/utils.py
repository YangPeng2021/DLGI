import torch
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

#change tensor to numpy
def ts_to_np(ts):
    
    return ts.detach().cpu().squeeze().numpy()

#change numpy to tensor
def np_to_ts(np):
    return torch.tensor(np).unsqueeze(0)

#calculate psnr
def batch_psnr(img, imgclean):
   
    psnr = 0
    for i in range(img.shape[0]):
        psnr += peak_signal_noise_ratio(imgclean[i,  :, :], img[i,:, :], data_range=1.)
    return psnr/img.shape[0]

def plot_loss_save(k,iters,y_loss,loss):
    #plt.cla()
    iters.append(k)
    loss_np=ts_to_np(loss)
    y_loss.append(loss_np) 

#plot loss function
def plot_loss(iters,y_loss):
    plt.plot(iters, y_loss, label='Loss rate', color='g')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('loss')
    plt.legend()   
    plt.pause(1.0)



