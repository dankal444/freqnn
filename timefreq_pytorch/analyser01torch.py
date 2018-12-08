# network computing regression, calculating frequency of a sine with 0 phase shift

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.autograd
from tensorboardX import SummaryWriter

MOMENTUM = 0.5

LEARNING_RATE = 0.001

SEED = 0
TRAIN_SIZE = 20000
DEV_SIZE = 5000
TEST_SIZE = 10000

DURATION = 0.1       # [s]
FS = 16000           # [Hz]
N_SAMPLES = int(DURATION * FS)
TWO_PI = 2 * np.pi;

# network params
N_TRAINING_EPOCHS = 300
BATCH_SIZE = 64

SEED = 0

RANDOM_STATE = np.random.RandomState(SEED)

A_BASE = 440
MIN_FREQ = A_BASE / 8  # A1
MAX_FREQ = A_BASE * 2 * 2 ** (5 / 12)  # D6
N_FREQS = int(np.round(np.log(MAX_FREQ / MIN_FREQ) / np.log(2 ** (1/96)))) + 1
exponents = np.arange(N_FREQS) / 96

FREQ_DOMAIN = MIN_FREQ * 2 ** exponents

stop = 1

# def create_test_sample(random_state=None):
#     freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * random_state.rand()
#     start_phase = random_state.rand() * TWO_PI
#     signal_x = np.linspace(0, DURATION, N_SAMPLES)
#     signal_y = np.sin(TWO_PI * freq * signal_x + start_phase)
#
#     return signal_y.astype(np.float32), np.float32(np.log(freq))


def create_train_sample(random_state=None):
    # freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * random_state.rand()
    freq_idx = random_state.randint(N_FREQS)
    freq = FREQ_DOMAIN[freq_idx]
    start_phase = random_state.rand() * TWO_PI
    signal_x = np.linspace(0, DURATION, N_SAMPLES)
    signal_y = np.sin(TWO_PI * freq * signal_x + start_phase)
    expected_result = np.zeros(N_FREQS)
    expected_result[freq_idx] = 1
    return signal_y.astype(np.float32), expected_result.astype(np.float32), freq


def create_test_sample(random_state=None):
    freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * random_state.rand()
    start_phase = random_state.rand() * TWO_PI
    signal_x = np.linspace(0, DURATION, N_SAMPLES)
    signal_y = np.sin(TWO_PI * freq * signal_x + start_phase)

    return signal_y.astype(np.float32), freq


class Analyser01Net(nn.Module):
    def __init__(self):
        super(Analyser01Net, self).__init__()
        # self.fft1_size = 4096
        self.linear1 = nn.Linear(N_SAMPLES + 2, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.final_layer = nn.Linear(1024, N_FREQS)

    def forward(self, input):
        input = input.float()
        x = torch.rfft(input, 1)
        # x = x[:, :, :int(x.shape[2] / 2)]
        x = x.view(-1, N_SAMPLES + 2)
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        output = self.final_layer(x)

        return output


class SineDatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        pass
        # self.min_freq = min_freq
        # self.max_freq = max_freq

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        sine_signal, expected_result, freq = create_train_sample(RANDOM_STATE)
        return sine_signal.reshape(1, -1), expected_result, np.array(freq).reshape(1)


class SineDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        # self.min_freq = min_freq
        # self.max_freq = max_freq
        pass

    def __len__(self):
        return 3000

    def __getitem__(self, item):
        sine_signal, freq = create_test_sample(RANDOM_STATE)
        return sine_signal.reshape(1, -1), np.array(freq).reshape(1)


if __name__ == '__main__':
    do_cuda = True
    writer = SummaryWriter('runs01')

    model = Analyser01Net()
    if do_cuda:
        model.cuda()

    train_loader = torch.utils.data.DataLoader(
        SineDatasetTrain(), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        SineDatasetTest(), batch_size=BATCH_SIZE, shuffle=True
    )

    def train(epoch):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE * 0.99 ** epoch, momentum=MOMENTUM)
        for batch_idx, (data, target, target_freq) in enumerate(train_loader):

            data = torch.autograd.Variable(data)
            target = torch.autograd.Variable(target)
            if do_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)

            output_numpy = output.cpu().detach().numpy()
            chosen_freq = FREQ_DOMAIN[np.argmax(output_numpy, axis=-1)]
            target_freq_numpy = target_freq.detach().numpy()
            halftone_freq_loss = np.mean(np.abs(np.log(chosen_freq / target_freq_numpy.flatten())) / np.log(2 ** (1/12)))

            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
                print('HalfTone freq loss: {:.4f}'.format(halftone_freq_loss))
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Loss', loss.data[0], niter)


    def test():
        model.eval()
        test_loss = 0
        n_batches = 0
        halftone_freq_loss = 0
        for data, target_freq in test_loader:
            data = torch.autograd.Variable(data, volatile=True)

            if do_cuda:
                data = data.cuda()

            output = model(data)
            output_numpy = output.cpu().detach().numpy()
            chosen_freq = FREQ_DOMAIN[np.argmax(output_numpy, axis=-1)]
            target_freq_numpy = target_freq.detach().numpy()
            halftone_freq_loss += np.mean(np.abs(np.log(chosen_freq / target_freq_numpy.flatten())) / np.log(2 ** (1/12)))
            n_batches += 1

        print('\nTest set: Average halftone loss: {:.6f}'.format(halftone_freq_loss / n_batches))
        # print('\nTest set: Average loss: {:.6f}'.format(test_loss))
        # writer.add_scalar('Test/Loss', loss.data[0], niter)

    for epoch in range(1, N_TRAINING_EPOCHS + 1):
        train(epoch)
        test()