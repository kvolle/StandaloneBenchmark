import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class PlottingUtils():
    def __init__(self, trained_model, viz_size=4, data=None, labels=None, normalization_used=None):
        self.trained_model = trained_model
        self.viz_size = viz_size
        self.data = data
        self.labels = labels
        self.gray_weights = np.array([0.299, 0.587, 0.114])#.reshape((1, 3))
        idx = np.arange(64)
        self.X, self.Y = np.meshgrid(idx, idx)
        if normalization_used is None:
            self.mean = [0., 0., 0.]
            self.std  = [1., 1., 1.]
        else:
            self.mean = normalization_used['mean']
            self.std = normalization_used['std']

    def denormalize_tensor(self, x, mean=None, std=None):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    def denormalize(self, x, mean=None, std=None):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        if len(x.shape) == 3:
            x = np.expand_dims(np.copy(x), 0)
            single_image = True
        # 3, H, W, B
        ten = np.moveaxis(np.copy(x), [0,3], [3,0])
        ten2 = np.copy(ten)
        for i, [t, m, s] in enumerate(zip(ten, mean, std)):
            #t.mul_(s).add_(m)
            ten2[i,:,:,:] = t*s + m
        # B, 3, H, W
        new_img = np.moveaxis(np.clip(ten2, 0, 1), [0, 3], [3, 0])
        if single_image:
            return np.squeeze(new_img)
        else:
            return new_img

    def latent_cloud(self, fig = None, dim = 2):
        if fig is None:
            fig = plt.figure(figsize=(10,8))
        colors = sns.color_palette('pastel')
    
        torch.manual_seed(42)
        idx = torch.randperm(len(self.data))
        with torch.no_grad():
            mu = self.trained_model.encoder(self.data).embedding.detach().cpu()
        if self.labels is not None:
            plt.scatter(mu[:, 0], mu[:, 1], c=self.labels, cmap=matplotlib.colors.ListedColormap(colors))

            cb = plt.colorbar()
            loc = np.arange(0,max(self.labels),max(self.labels)/float(len(colors)))
            cb.set_ticks(loc)
            cb.set_ticklabels([f'{i}' for i in range(10)])
        else:
            plt.scatter(mu[:, 0], mu[:,1])
            
        plt.tight_layout()
        return fig
    
    def grid_images(self, axes, data):
        
        data = data.clone().detach().cpu().numpy()
        for i in range(self.viz_size):
            for j in range(self.viz_size):
                #gen_ = gen_data[i*self.viz_size +j].cpu().numpy()
                data_ = data[i*self.viz_size + j, :, :, :]
                if data_.shape[0] == 1:
                    data_ = data_.squeeze(0)
                    color_map = 'gray'
                else:
                    data_ = self.denormalize(np.moveaxis(data_, 0, 2))
                    color_map = None
                axes[i][j].imshow(data_, cmap=color_map)
                axes[i][j].axis('off')
        plt.tight_layout(pad=0.)

    def grid_hists(self, axes, data):
        
        for i in range(self.viz_size, data):
            data = data.clone().detach().cpu().numpy()
            for j in range(self.viz_size):
                data_ = data[i*self.viz_size + j, :, :, :]
                if data_.shape[0] == 1:
                    data_ = data_.squeeze(0)
                    axes[i, j].hist(data_.flatten(), histtype='step')
                else:
                    data_ = np.moveaxis(data_, 0, 2)
                    axes[i][j].hist(data_[:,:,0].flatten(),histtype='step',color='r')
                    axes[i][j].hist(data_[:,:,1].flatten(),histtype='step',color='g')
                    axes[i][j].hist(data_[:,:,2].flatten(),histtype='step',color='b')
                axes[i][j].set_xlim((0., 1.))
        plt.tight_layout(pad=0.)

    def grid_gray_heights(self, axes, data):
        data = data.clone().detach().cpu().numpy()
        for i in range(self.viz_size):
            for j in range(self.viz_size):
                data_ = np.moveaxis(data[i*self.viz_size + j, :, :, :],0, -1)
                data_ = np.inner(data_, self.gray_weights)
                axes[i][j].plot_surface(self.X, self.Y, np.transpose(data_), cmap='viridis', linewidth=0)
                axes[i][j].axis('off')
        plt.tight_layout(pad=0.)

    def generated_from_samples(self, gen_data, hist = False):
        # show results with normal sampler
        fig, axes = plt.subplots(nrows=self.viz_size, ncols=self.viz_size, figsize=(10, 10))
        fig.canvas.manager.set_window_title('Normal Sampler')
        if hist:
            self.grid_hists(axes, gen_data)
        else:
            self.grid_images(axes, gen_data)

    def reconstructed(self, plot_type = "img"):
        reconstructions = self.trained_model.reconstruct(self.data[:self.viz_size**2]).detach().cpu()
        # show reconstructions
        fig, axes = plt.subplots(nrows=self.viz_size, ncols=self.viz_size, figsize=(10, 10))
        fig.canvas.manager.set_window_title('Reconstructions')

        if plot_type == "hist":
            self.grid_hists(axes, reconstructions)
        elif plot_type == "height":
            plt.close(fig)
            fig, axes = plt.subplots(nrows=self.viz_size, ncols=self.viz_size, figsize=(10, 10), subplot_kw={'projection':'3d'})
            fig.canvas.manager.set_window_title('Reconstructions')
            self.grid_gray_heights(axes, reconstructions)
        elif plot_type == "img":
            self.grid_images(axes, reconstructions)
        else:
            print("Plot type not recognized")

    def original(self, plot_type = "img"):
        # show reconstructions
        fig, axes = plt.subplots(nrows=self.viz_size, ncols=self.viz_size, figsize=(10, 10))
        fig.canvas.manager.set_window_title('Original Images')

        if plot_type == "hist":
            self.grid_hists(axes, self.data)
        elif plot_type == "height":
            plt.close(fig)
            fig, axes = plt.subplots(nrows=self.viz_size, ncols=self.viz_size, figsize=(10, 10), subplot_kw={'projection':'3d'})
            fig.canvas.manager.set_window_title('Original Images')
            self.grid_gray_heights(axes, self.data)
        elif plot_type == "img":
            self.grid_images(axes, self.data)
        else:
            print("Plot type not recognized")

    def interpolations(self):
        # Visualizing Interpolations
        interim_steps = 10
        interpolations = self.trained_model.interpolate(self.data[:self.viz_size], self.data[self.viz_size:2*self.viz_size], granularity=interim_steps).detach().cpu()
        
        # show interpolations
        fig, axes = plt.subplots(nrows=self.viz_size, ncols=interim_steps, figsize=(10, 5))
        fig.canvas.manager.set_window_title('Interpolation Test')
        for i in range(self.viz_size):
            for j in range(interim_steps):
                interp_ = interpolations[i, j].cpu().numpy()
                if interp_.shape[0] == 1:
                    interp_ = interp_.squeeze(0)
                    color_map = 'gray'
                else:
                    interp_ = self.denormalize(np.moveaxis(interp_, 0, 2))
                    color_map = None
                axes[i][j].imshow(interp_, cmap=color_map)
                axes[i][j].axis('off')
        plt.tight_layout(pad=0.)

    def show(self):
        plt.show()