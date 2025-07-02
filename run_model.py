"""
Copyright © 2025 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pyloudnorm as pyln
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
import math
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ############################
# #### LOAD S3QA MODEL ####
# ############################

# function to initialize model weights (called by the hltcoe xvec github code)
def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):

            # logger.info("Initializing %s with kaiming normal" % str(m))
            nn.init.kaiming_normal_(m.weight, a=0.01)  # default negative slope of Leakygelu
        elif isinstance(m, nn.BatchNorm1d):
            if m.affine:
                # logger.info("Initializing %s with constant (1,. 0)" % str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# https://github.com/Lightning-AI/lightning/discussions/14377
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# inspired by wav2vec/hubert with some design choices informed by Coon & Punjabi
class s3qa_Model(pl.LightningModule):
    def __init__(self, nclasses: int = 1, batch_size: int = 16, lr: float = 0.0001, dur_aug=False):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.lr = lr
        self.nclasses = nclasses
        self.eps = 1e-9
        self.dur_aug = dur_aug

        self.embedding_dim = 384
        self.proj = 128
        self.initial_channels = 128

        self.conv_enc0 = torch.nn.Conv1d(1, self.initial_channels, kernel_size=10, stride=5)

        self.layer_norm0 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_enc1 = torch.nn.Conv1d(self.initial_channels, self.initial_channels, kernel_size=3, stride=2)
        self.layer_norm1 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_enc2 = torch.nn.Conv1d(self.initial_channels, self.initial_channels, kernel_size=3, stride=2)
        self.layer_norm2 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_enc3 = torch.nn.Conv1d(self.initial_channels, self.initial_channels, kernel_size=3, stride=2)
        self.layer_norm3 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_enc4 = torch.nn.Conv1d(self.initial_channels, self.initial_channels, kernel_size=3, stride=2)
        self.layer_norm4 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_enc5 = torch.nn.Conv1d(self.initial_channels, self.initial_channels, kernel_size=2, stride=2)
        self.layer_norm5 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_enc6 = torch.nn.Conv1d(self.initial_channels, self.initial_channels, kernel_size=2, stride=2)
        self.layer_norm6 = torch.nn.LayerNorm(self.initial_channels)

        self.conv_embed = torch.nn.Linear(self.initial_channels, self.embedding_dim)

        self.pos_encoder = PositionalEncoding(d_model=self.embedding_dim, dropout=0.0)

        self.trans_enc_layer = torch.nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, activation="gelu",
                                                                dropout=0.05, dim_feedforward=1536)
        self.trans_enc = torch.nn.TransformerEncoder(self.trans_enc_layer, num_layers=6)

        self.transavg = torch.nn.AdaptiveAvgPool1d(1)

        self.embed_avg2out = torch.nn.Linear(self.embedding_dim, self.proj)

        self.out_logits = torch.nn.Linear(self.proj, self.nclasses)
        init_weight(self)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.gelu(self.layer_norm0(self.conv_enc0(x.float()).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.gelu(self.layer_norm1(self.conv_enc1(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.gelu(self.layer_norm2(self.conv_enc2(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.gelu(self.layer_norm3(self.conv_enc3(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.gelu(self.layer_norm4(self.conv_enc4(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.gelu(self.layer_norm5(self.conv_enc5(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = F.gelu(self.layer_norm6(self.conv_enc6(x).permute(0, 2, 1)).permute(0, 2, 1))

        x_conv_enc_out = x

        x = x.permute(2, 0, 1)

        x = F.gelu(self.conv_embed(x))

        conv_embed_out = x

        x = self.pos_encoder(x)

        x = self.trans_enc(x)

        x_trans_enc_out = x

        x = x.permute(1, 2, 0)

        x = self.transavg(x)

        x = x.permute(0, 2, 1)
        x_avg_out = x

        x = F.gelu(self.embed_avg2out(x))
        x_embed = x

        out_logit = self.out_logits(x_embed)

        out_sigmoid = self.sigmoid(out_logit)

        return out_logit, out_sigmoid, x_embed, x_avg_out, x_trans_enc_out, conv_embed_out, x_conv_enc_out


model_ckpt_path = sys.argv[1]
s3qa_checkpoint = torch.load(model_ckpt_path, map_location=device)

dur_aug = False
lr = 0.0005
batch_size = 1

s3qa_model = s3qa_Model(nclasses=1, batch_size=batch_size, lr=lr, dur_aug=dur_aug)
print(s3qa_model)
s3qa_model.load_state_dict(s3qa_checkpoint['state_dict'])
print(s3qa_model)
s3qa_model.eval()
print(ModelSummary(s3qa_model))

###########################
#### LOAD, SET UP DATA ####
###########################


data_path = sys.argv[2]
if not data_path.endswith(os.path.sep):
    data_path += os.path.sep

print('starting: ', data_path)
print('collecting audio...')

master_out_path = data_path[:-1] + '.csv'
audio_files = []
for xx in Path(data_path).glob("**/*.wav"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.mp3"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.flac"):
    audio_files.append(xx)

print('collected audio files: ', str(len(audio_files)))

seg_dur_s = 4
seg_hop_s = 1

initialize_master_arr = True

##################################
#### KICK OFF INFERENCE ####
##################################

for idx, f in enumerate(tqdm(audio_files)):

    print('Starting! ' + str(f))

    mod_speech, m_sr = torchaudio.load(f)

    assert m_sr == 16000

    # normalize the full file, but the embeddings and ASR models will have their own normalization routines
    # will want to re-norm any 4s segment that goes into the model
    meter = pyln.Meter(m_sr)
    mod_speech_np = torch.squeeze(mod_speech).numpy()
    print(mod_speech_np.shape)
    seg_loudness = meter.integrated_loudness(mod_speech_np)
    mod_speech_np = pyln.normalize.loudness(mod_speech_np, seg_loudness, -35.0)
    mod_speech = torch.unsqueeze(torch.tensor(mod_speech_np), 0)
    print(mod_speech.shape)

    print('mod_speech shape: ', mod_speech.shape)

    outputs = []
    if mod_speech.shape[-1] <= (m_sr * 4):
        with torch.no_grad():
            out_logit, out_sigmoid, x_embed, x_avg_out, x_trans_enc_out, conv_embed_out, x_conv_enc_out = s3qa_model(
                mod_speech.unsqueeze(0).float())
        print(out_logit.shape)
        y_hat_logits = torch.squeeze(out_logit, 1)
        y_hat_logits = torch.squeeze(y_hat_logits, 1)
        outputs.append(y_hat_logits.detach().numpy()[0])
    else:
        print(mod_speech.shape)
        n_segs = int(np.ceil((mod_speech.shape[-1] - (m_sr * (seg_dur_s - seg_hop_s))) / (m_sr * seg_hop_s)))

        seg_list = list(range(n_segs))

        sn = 0
        next_end_idx = 0
        while next_end_idx <= mod_speech.shape[-1]:
            start_idx = (seg_hop_s * m_sr) * sn
            # print(start_idx)
            end_idx = start_idx + (m_sr * seg_dur_s)
            # print(end_idx)
            next_end_idx = end_idx + (seg_hop_s * m_sr)
            this_seg = mod_speech[:, int(start_idx):int(end_idx)]
            sn += 1

            seg_meter = pyln.Meter(m_sr)
            this_seg_np = torch.squeeze(this_seg).numpy()
            print(this_seg_np.shape)
            seg_loudness = seg_meter.integrated_loudness(this_seg_np)
            this_seg_np = pyln.normalize.loudness(this_seg_np, seg_loudness, -35.0)
            print(this_seg_np.shape)
            this_seg = torch.unsqueeze(torch.tensor(this_seg_np), 0)
            print(this_seg.shape)

            with torch.no_grad():
                out_logit, out_sigmoid, x_embed, x_avg_out, x_trans_enc_out, conv_embed_out, x_conv_enc_out = s3qa_model(
                    this_seg.unsqueeze(0).float())
            print(out_logit.shape)
            y_hat_logits = torch.squeeze(out_logit, 1)
            y_hat_logits = torch.squeeze(y_hat_logits, 1)
            outputs.append(y_hat_logits.detach().numpy()[0])

        start_idx = (seg_hop_s * m_sr) * sn
        # print(start_idx)
        end_idx = mod_speech.shape[-1]
        # print(end_idx)
        this_seg = mod_speech[:, int(start_idx):int(end_idx)]

        seg_meter = pyln.Meter(m_sr)
        this_seg_np = torch.squeeze(this_seg).numpy()
        print(this_seg_np.shape)
        seg_loudness = seg_meter.integrated_loudness(this_seg_np)
        this_seg_np = pyln.normalize.loudness(this_seg_np, seg_loudness, -35.0)
        this_seg = torch.unsqueeze(torch.tensor(this_seg_np), 0)
        print(this_seg.shape)

        with torch.no_grad():
            out_logit, out_sigmoid, x_embed, x_avg_out, x_trans_enc_out, conv_embed_out, x_conv_enc_out = s3qa_model(
                this_seg.unsqueeze(0).float())
        print(out_logit.shape)
        y_hat_logits = torch.squeeze(out_logit, 1)
        y_hat_logits = torch.squeeze(y_hat_logits, 1)
        outputs.append(y_hat_logits.detach().numpy()[0])

    print(str(mod_speech.shape[-1] / m_sr), outputs)
    s3qa_output_mean = np.mean(outputs)
    s3qa_output_median = np.median(outputs)
    print(s3qa_output_mean)

    output_txt = {
        'fpath': [f],
        's3qa_output_mean': [s3qa_output_mean],
        's3qa_output_median': [s3qa_output_median],
        'all_outputs_at_hop': [outputs]
    }

    if not os.path.exists(master_out_path) or initialize_master_arr:
        pd.DataFrame(output_txt).to_csv(master_out_path, index=False, header=True, mode='w')
        initialize_master_arr = False

    else:
        pd.DataFrame(output_txt).to_csv(master_out_path, index=False, header=False, mode='a')

    print('done! ', str(f))
