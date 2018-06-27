from __future__ import print_function
from __future__ import division

import argparse

import matplotlib.pyplot as plot
import numpy as np
import torch.utils.data
import torchvision.datasets
from torch import nn, optim
from torchvision import transforms


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = tuple(shape)

    def forward(self, input):
        return input.view(len(input), *self.shape)

    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, self.shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(len(input), int(np.prod(input.shape[1:])))


class AddNoise(nn.Module):
    def __init__(self, mean=0.0, stddev=0.1):
        super(AddNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, input):
        noise = torch.empty_like(input).normal_(self.mean, self.stddev)
        return input + noise


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(noise_size, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256, momentum=0.9),
            nn.LeakyReLU(0.2),
            Reshape(256, 7, 7),
            # Layer 2
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            # Layer 3
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            # Layer 4
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.2),
            # Output
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            AddNoise(),
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            # Output
            Flatten(),
            nn.Linear(256 * 2 * 2, 1),
        )

    def forward(self, images):
        return self.model(images)


parser = argparse.ArgumentParser(description="PyTorch LSGAN")
parser.add_argument("-n", "--noise-size", type=int, default=100)
parser.add_argument("-e", "--epochs", type=int, default=30)
parser.add_argument("-b", "--batch-size", type=int, default=256)
parser.add_argument("-x", "--examples", type=int, default=8)
parser.add_argument("-s", "--show", action="store_true")
parser.add_argument("-g", "--cuda", action="store_true")
options = parser.parse_args()

generator = Generator(options.noise_size)
discriminator = Discriminator()

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

criterion = nn.MSELoss()

discriminator_optimizer = optim.Adam(
    discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999)
)
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

print(discriminator)
print(generator)

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=options.batch_size, num_workers=4, shuffle=True, drop_last=True
)

device = torch.device("cuda:0" if options.cuda else "cpu")
generator.to(device)
discriminator.to(device)
criterion.to(device)

try:
    for epoch_index in range(1, options.epochs + 1):
        for batch_index, batch in enumerate(data_loader):
            discriminator.zero_grad()
            real_images = batch[0].to(device)
            real_predictions = discriminator(real_images)
            real_labels = torch.empty(options.batch_size, 1, device=device)
            real_labels = real_labels.uniform_(0.8, 1.0)
            d_loss_real = criterion(real_predictions, real_labels)
            d_loss_real.backward()

            noise = torch.randn(options.batch_size, options.noise_size, device=device)
            fake_images = generator(noise)
            fake_predictions = discriminator(fake_images.detach())
            fake_labels = torch.zeros(options.batch_size, 1, device=device)
            d_loss_fake = criterion(fake_predictions, fake_labels)
            d_loss_fake.backward()

            d_loss_value = d_loss_real.mean() + d_loss_fake.mean()
            discriminator_optimizer.step()

            generator.zero_grad()
            fake_labels.fill_(1)
            fake_predictions = discriminator(fake_images)
            g_loss = criterion(fake_predictions, fake_labels)
            g_loss.backward()
            g_loss_value = g_loss.mean()
            generator_optimizer.step()

            message = "Epoch: {}/{} | Batch: {}/{} | G: {:.10f} D: {:.10f}"
            message = message.format(
                epoch_index,
                options.epochs,
                batch_index,
                len(data_loader),
                g_loss_value,
                d_loss_value,
            )
            print("\r{}".format(message), end="")
        print()
except KeyboardInterrupt:
    pass

print("\nTraining complete!")

if not options.show:
    plot.switch_backend("Agg")

generator.eval()

noise = torch.empty(options.examples, options.noise_size)
if options.cuda:
    noise = noise.cuda()
images = generator(noise).detach().cpu().numpy()
images = (images + 1) / 2

number_of_columns = 4
number_of_rows = options.examples // number_of_columns
plot.figure()
for i, image in enumerate(images):
    plot.subplot(number_of_rows, number_of_columns, 1 + i)
    plot.axis("off")
    plot.imshow(image.squeeze(), cmap="gray")

if options.show:
    print("Showing plots")
    plot.show()
else:
    print("Saving figure.png")
    plot.savefig("figure.png")
