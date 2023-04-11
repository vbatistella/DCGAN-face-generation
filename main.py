from __future__ import print_function
#%matplotlib inline
from models.generator import Generator
from models.discriminator import Discriminator

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

#
# Parameters
#
data_path = "./data/"
batch_size = 128
image_size = 64
num_epochs = 10


# Initiate weights as normal
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset():
    workers = 2
    dataset = dset.ImageFolder(root=data_path, transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

def train():
    #
    # CONSTANTS
    #
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    lr = 0.0002
    beta1 = 0.5

    # manualSeed = 999
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = get_device()
    dataloader = get_dataset()

    # Create the generator
    netG = Generator(nz, ngf, nc).to(device)
    # Initalize generator weights
    netG.apply(weights_init)
    # Print the model
    print(netG)

    # Create the discriminator
    netD = Discriminator(nc, ndf).to(device)
    # Initalize discriminator weights
    netD.apply(weights_init)
    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Setup Adam optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # Latent Vector
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Progress lists
    img_list = []
    G_losses = []
    D_losses = []

    # Convention labels
    real_label = 1.
    fake_label = 0.

    print("Starting Training Loop...")
    iters = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            #
            # Update G
            #
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print("Epoch %d - D Loss: %.3f, \tG Loss: %.3f | D(x): %.3f \tD(G(z)): %.3f / %.3f"
                      % (epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                )
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

if __name__ == "__main__":
    train()