from fastai.vision import *
from fastai.callbacks.hooks import *
from PIL import Image

def generate_cam(learn, path, cl):
    # GradCAM logic
    m = learn.model.eval()
    im = open_image(path)
    min_size = im.shape[1] if im.shape[1] < im.shape[2] else im.shape[2]
    im = im.crop((min_size, min_size)).resize(640)
    xb,_ = learn.data.one_item(im)
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cl - 1)].backward()
    acts = hook_a.stored[0].cpu()
    grad = hook_g.stored[0][0].cpu()
    grad_chan = grad.mean(1).mean(1)
    mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
    
    # Visualize results
    xb_im = xb[0]    
    sz = list(xb_im.shape[-2:])
    fig = plt.figure(frameon=False, figsize=(20,10))
    plt.figure
    plt.tight_layout()
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title('Plate '+str(cl)+' Grad-CAM explanation')
    im.show(ax1)
    ax1.imshow(mult, alpha=0.7, extent=(0,*sz[::-1],0), interpolation='bicubic', cmap='magma')
    ax1.set_axis_off()
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.set_title('Plate '+str(cl))
    im.show(ax2)