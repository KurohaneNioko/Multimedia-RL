import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchaudioclip import stft_and_db


class BandoriMusicNet(nn.Module):
    def __init__(self, input_size, layer_num, mid_size=256, output_size=256, device='cuda'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(input_size, mid_size), nn.ReLU(inplace=True),
        ]
        for _ in range(layer_num):
            self.model += [nn.Linear(mid_size, mid_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(mid_size, output_size)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float32)
        batch = s.shape[0]
        s = s.view(batch, -1)
        music_out = self.model(s)
        return music_out


class PreBandoriPPO(nn.Module):
    def __init__(self, is_actor, input_channel=3, interval=3,  # height=72,
                 audio_state=True, audio_input_size=129 * 37, audio_net_layer=2, audio_mid_size=256,
                 audio_output_size=256,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.input_channel = input_channel
        self.interval = interval
        self._is_actor = is_actor
        # self.one_frame_shape = height>>1, (height<<4)//9  # height is attr of full screen
        self.one_frame_shape = 23, 45  # height is attr of full screen
        # music model
        self.audio_state = audio_state
        if self.audio_state:
            self.music_model = BandoriMusicNet(
                audio_input_size, audio_net_layer, audio_mid_size, audio_output_size, device)
        # graph model, input shape = [batch, input_channel, interval, height_ width]
        self.conv1 = nn.Conv3d(in_channels=self.input_channel, out_channels=32, kernel_size=(1, 3, 6),
                               stride=(1, 1, 2))
        self.relu1 = nn.ReLU(inplace=True)
        # input = 32, 3, 21, 20
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        # input = 64, 3, 10, 9
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3))
        self.relu3 = nn.ReLU(inplace=True)  # 64, 3, 8, 7
        # graph: FC -> 256
        self.fc_graph = self.fc_graph = nn.Linear(10752, 256).to(self.device)
        self.total_relu = nn.ReLU(inplace=True)
        self.total_FC = nn.Linear(512 if self.audio_state else 256, 3 if self._is_actor else 1)

    def forward(self, s, state=None, info={}):
        # s.shape = [batch, len([height, width, channel]*3, audio_shape) ]
        batch = s.shape[0] if len(s.shape)!=1 else 1
        # if batch == 1: s = s[np.newaxis, :]
        if len(s.shape) == 1: s = s[np.newaxis, :]
        graph_batch = torch.tensor(list(s[:, 0:self.interval]), dtype=torch.float32).to(self.device)
        # if not isinstance(s, torch.Tensor):
        #     s = torch.tensor(s, device=self.device, dtype=torch.float32)
        # shape=[batch, 3-png-shape] -> batch, channel, interval, height, width, verified in small rand tensor
        graph_batch = (graph_batch / 255.).squeeze(dim=0).view(
            [batch, self.input_channel, self.interval, *self.one_frame_shape])
        x = self.conv1(graph_batch)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(batch, -1)
        # print(int(torch.tensor(x[0].shape, dtype=torch.int).prod()))
        # self.fc_graph = nn.Linear(int(torch.tensor(x[0].shape, dtype=torch.int).prod()), 256).to(self.device)
        x = self.fc_graph(x)
        if self.audio_state:
            audio_input = stft_and_db(list(s[:, -1]), n_fft=256)
            y = self.music_model(audio_input)
            x = torch.cat((x, y), dim=-1)
        x = self.total_relu(x)
        logits = self.total_FC(x)
        if self._is_actor:
            return F.softmax(logits, dim=-1), state
        else:
            return logits

# (23, 45, 3) torch.Size([129, 37])
