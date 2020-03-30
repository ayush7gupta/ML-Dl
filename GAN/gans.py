import os
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from GAN.util.utils import Logger

def load_mnist_data():
  """
  Function to load mnist data
  """
  compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
  out_dir = './dataset'
  return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def array_to_vector(array):
  """
  Convert array to vector
  """
  return array.view(array.size(0), 784)

def vector_to_array(vector):
  """
  Convert vector to array
  """
  return vector.view(vector.size(0), 1, 28, 28)


data = load_mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

# Num batches
num_batches = len(data_loader)

print(f'Number of batches - {num_batches}')
print(f'Number of samples - {len(data)}')


class DiscriminatorNet(torch.nn.Module):
  """
  A three hidden-layer discriminative neural network
  """

  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    n_features = 784
    n_out = 1

    self.hidden0 = nn.Sequential(
      nn.Linear(n_features, 1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.hidden1 = nn.Sequential(
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.hidden2 = nn.Sequential(
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.out = nn.Sequential(
      torch.nn.Linear(256, n_out),
      torch.nn.Sigmoid()
    )

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x


discriminator = DiscriminatorNet()

if (torch.cuda.is_available):
  discriminator = discriminator.cuda()

# print architecture
print(discriminator)


# Generator
class GeneratorNet(torch.nn.Module):
  """
  A three hidden-layer generative neural network
  """

  def __init__(self):
    super(GeneratorNet, self).__init__()
    n_features = 100
    n_out = 784

    self.hidden0 = nn.Sequential(
      nn.Linear(n_features, 256),
      nn.LeakyReLU(0.2)
    )
    self.hidden1 = nn.Sequential(
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2)
    )
    self.hidden2 = nn.Sequential(
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2)
    )

    self.out = nn.Sequential(
      nn.Linear(1024, n_out),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x


generator = GeneratorNet()

if (torch.cuda.is_available):
  generator = generator.cuda()

# print architecture
print(generator)

#Random noise sampler
def noise(size):
  """
  Generates a 1-d vector of gaussian sampled random values
  """
  z = Variable(torch.randn(size, 100))
  return z

#Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

#Target vectors
def ones_target(size):
  """
  Tensor containing ones, with shape = size
  """
  data = Variable(torch.ones(size, 1))
  return data

def zeros_target(size):
  '''
  Tensor containing zeros, with shape = size
  '''
  data = Variable(torch.zeros(size, 1))
  return data


# Discriminator Trainer
def train_discriminator(optimizer, real_data, fake_data):
  N = real_data.size(0)

  # Reset gradients
  optimizer.zero_grad()

  # Train on Real Data
  prediction_real = discriminator(real_data.cuda())
  prediction_real = prediction_real.cuda()

  # Calculate error and backpropagate
  error_real = loss(prediction_real.cuda(), ones_target(N).cuda())
  error_real = error_real.cuda()
  error_real.backward()

  # Train on Fake Data
  prediction_fake = discriminator(fake_data.cuda())
  prediction_fake = prediction_fake.cuda()

  # Calculate error and backpropagate
  error_fake = loss(prediction_fake.cuda(), zeros_target(N).cuda())
  error_fake = error_fake.cuda()
  error_fake.backward()

  # Update weights with gradients
  optimizer.step()

  # Return error and predictions for real and fake inputs
  return error_real + error_fake, prediction_real, prediction_fake

#Generator Trainer
def train_generator(optimizer, fake_data):
  N = fake_data.size(0)

  # Reset gradients
  optimizer.zero_grad()

  # Sample noise and generate fake data
  prediction = discriminator(fake_data.cuda())
  prediction = prediction.cuda()

  # Calculate error and backpropagate
  error = loss(prediction.cuda(), ones_target(N).cuda())
  error = error.cuda()
  error.backward()

  # Update weights with gradients
  optimizer.step()

  # Return error
  return error

#Testing

num_test_samples = 16
test_noise = noise(num_test_samples)

#Training
# Create logger instance
logger = Logger(model_name='Vanilla_GAN', data_name='MNIST')

# Total number of epochs to train
num_epochs = 200

for epoch in range(num_epochs):
  for n_batch, (real_batch,_) in enumerate(data_loader):
    N = real_batch.size(0)

    # Train Discriminator
    real_data = Variable(array_to_vector(real_batch).cuda())

    # Generate fake data and detach
    fake_data = generator((noise(N).cuda().detach()))

    # Train D
    d_error, d_pred_real, d_pred_fake = \
          train_discriminator(d_optimizer, real_data.cuda(), fake_data.cuda())

    # Train Generator
    # Generate fake data
    fake_data = generator(noise(N).cuda())

    # Train G
    g_error = train_generator(g_optimizer, fake_data)

    # Log batch error
    logger.log(d_error, g_error, epoch, n_batch, num_batches)

    # Display Progress every few batches
    if (n_batch) % 100 == 0:
        test_images = vector_to_array(generator(test_noise.cuda()))
        test_images = test_images.data

        logger.log_images(
            test_images.cpu(), num_test_samples,
            epoch, n_batch, num_batches
        );
        # Display status Logs
        logger.display_status(
            epoch, num_epochs, n_batch, num_batches,
            d_error, g_error, d_pred_real, d_pred_fake
        )