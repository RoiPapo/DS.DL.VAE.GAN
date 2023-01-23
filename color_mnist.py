import random
from typing import Optional
from random import randint
import torch
from PIL import Image
import PIL
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import numpy as np
from torch.optim import Adam
import visualize
from Trainer import Trainer
from torch.autograd import Variable
from visualize import Visualizer as Viz
from torch import optim
from typing import Callable, Optional

colors = {
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 128, 0],
    4: [255, 0, 255],
    5: [128, 128, 128],
    6: [128, 0, 255],
    7: [0, 255, 255],
    8: [255, 255, 0],
    9: [255, 0, 128]
}

from models import VAE


class color(object):
    def __call__(self, img):
        """
        :param img: (PIL): Image
        :return: ycbr color space image (PIL)
        """
        img = np.asarray(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        chosen_color = colors[randint(0, 9)]
        img2 = np.zeros((32, 32, 3))
        img2[:, :, 0] = img * chosen_color[0]
        img2[:, :, 1] = img * chosen_color[1]
        img2[:, :, 2] = img * chosen_color[2]
        # img2 = np.transpose(img2, (1, 2, 0)).astype(int)

        return img2.astype(np.uint8)
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + '()'


def wrapper(x):
    class _wrap(MNIST):
        def __init__(
                self,
                root: str,
                train: bool,

        ) -> None:
            transforme = transforms.Compose([
                transforms.Resize(32),
                color(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
            ])
            download = True if train else False
            super().__init__(root, train, transforme, download=download)

    return _wrap


def plot_latent(autoencoder, data_loader, mode, num_batches=100):
    """
    Plotting the Autoencoder's latent space
    :param autoencoder: Autoencoder model
    :param data_loader:
    :param device: cuda/cpu
    :param mode: 'continuous' or 'discrete' - used in plot's title
    :param num_batches: #points to plot
    """
    for i, (x, y) in enumerate(data_loader):
        z = autoencoder.encode(Variable(x))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            plt.title(f"{mode.capitalize()} latent space")
            plt.show()
            break


if __name__ == '__main__':
    ColorMNISTDataset = wrapper(datasets.MNIST)
    train_ColorMNISTDataset = ColorMNISTDataset(root='./data/', train=True)
    test_ColorMNISTDataset = ColorMNISTDataset(root='./data/', train=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_ColorMNISTDataset,
                                               batch_size=64,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_ColorMNISTDataset,
                                              batch_size=64,
                                              shuffle=True)

    # Build a dataloader for your data
    # dataloader = get_my_dataloader(batch_size=32)
    dataloader = train_loader
    # Define latent distribution
    latent_spec_desc = {'disc': [10, 5, 5, 2]}
    latent_spec_cont = {'cont': 20}
    latent_spec_both = {'cont': 20, 'disc': [10, 5, 5, 2]}

    # Build a Joint-VAE model
    model_desc = VAE(img_size=(3, 32, 32), latent_spec=latent_spec_desc)
    model_cont = VAE(img_size=(3, 32, 32), latent_spec=latent_spec_cont)
    model_both = VAE(img_size=(3, 32, 32), latent_spec=latent_spec_both)

    # Build a trainer and train model
    labels = ["model_desc","model_cont","model_both"]
    Models= [model_desc, model_cont, model_both]
    for i,model in enumerate([model_both]):
        optimizer = Adam(model.parameters())
        trainer = Trainer(model, optimizer,
                          cont_capacity=[0., 5., 25000, 30.],
                          disc_capacity=[0., 5., 25000, 30.])
        trainer.train(dataloader, epochs=20)
        torch.save(model, "C:\\Users\\RoiPapo\\Downloads\\ModelsVAE"+labels[i]+".pt")

        #
        # # Visualize samples from the model
        # viz = Viz(model)
        # samples = viz.samples(filename=labels[i]+".png")
        # a = viz.latent_traversal_line()
        # b = viz.latent_traversal_grid()
        # c = viz.all_latent_traversals()
        #
        # traversals = viz.latent_traversal_grid(cont_idx=2, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
        # ordering = [9, 3, 0, 5, 7, 6, 4, 8, 1, 2]  # The 9th dimension corresponds to 0, the 3rd to 1 etc...
        # traversals = visualize.reorder_img(traversals, ordering, by_row=True)
        plot_latent(autoencoder=model,data_loader=dataloader,mode=labels[i])

        # plt.imshow(traversals.numpy())
