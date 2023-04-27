##import os
##path =r"C:\Users\HP\Downloads\VITS-Paimon\content\VITS-Paimon"
##os.chdir(path)
##print(os.getcwd())
##
###%matplotlib inline
import Cython
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
##from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
##from text import text_to_sequence
##from scipy.io.wavfile import write
##
##
##def get_text(text, hps):
##    text_norm = text_to_sequence(text, hps.data.text_cleaners)
##    if hps.data.add_blank:
##        text_norm = commons.intersperse(text_norm, 0)
##    text_norm = torch.LongTensor(text_norm)
##    return text_norm
##
import utils
hps = utils.get_hparams_from_file("./configs/biaobei_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint('G_1434000.pth', net_g, None)
import soundfile as sf
text = "\u6CE8\u610F\u770B\uFF0C\u8FD9\u4E2A\u7537\u4EBA\u53EB\u505A\u7A7A\uFF0C\u522B\u7728\u773C\uFF0C\u4ED6\u6B63\u5728\u53C2\u52A0\u4F5B\u94B5\u4E50\u8D85\u7EA7\u7279\u5DE5\u7684\u9762\u8BD5" #@param {type: 'string'}
length_scale = 1 #@param {type:"slider", min:0.1, max:3, step:0.05}
filename = 'test' #@param {type: "string"}
audio_path = f'{filename}.wav'
stn_tst = get_text(text, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))
sf.write(audio_path,samplerate=hps.data.sampling_rate)
