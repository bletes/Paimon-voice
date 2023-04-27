#改好路径，管理员模式运行后，复制一段文字即可-2022-11-26
#import IPython.display as ipd
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import soundfile as sf
import time
import pyperclip
#from pynput import keyboard
#from pynput.mouse import Listener
from playsound import playsound

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


"""# 语音合成

你还可以通过调节`length_scale`来控制说话的速度！注意`length_scale`默认值为1.0，值越大说话越**慢**。

**Tips**

1. 测试发现太长或太短的句子效果都不太好，参考文本建议分句使用。
2. 「、」的停顿效果不是很理想，建议使用「。」或「…」。
3. 由于数据集中去掉了大部分H片段，余下数据里各个人物说话的语调都比较平稳。因此使用「！」可能会出现破音。

为生合成的语音设定一个文件名。注意不需要加扩展名！

命名后运行该代码块，你将在左侧文件系统中/content/vits-mandarin-biaobei/test.wav找到它！
"""

hps = utils.get_hparams_from_file("./configs/biaobei_base.json")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model,device_ids=[0,1,2])


net_g = SynthesizerTrn(len(symbols),hps.data.filter_length // 2 + 1,hps.train.segment_size // hps.data.hop_length,**hps.model).to(device)
_ = net_g.eval()
_ = utils.load_checkpoint('G_1434000.pth', net_g, None)
text = "\u4E0B\u9762\u7ED9\u5927\u5BB6\u7B80\u5355\u4ECB\u7ECD\u4E00\u4E0B\u600E\u4E48\u4F7F\u7528\u8FD9\u4E2A\u6559\u7A0B\u5427\uFF01\u9996\u5148\u6211\u4EEC\u8981\u6709\u9B54\u6CD5\uFF0C\u624D\u80FD\u8BBF\u95EE\u5230\u8C37\u6B4C\u7684\u4E91\u5E73\u53F0\u3002\u70B9\u51FB\u8FDE\u63A5\u5E76\u66F4\u6539\u8FD0\u884C\u65F6\u7C7B\u578B\uFF0C\u8BBE\u7F6E\u786C\u4EF6\u52A0\u901F\u5668\u4E3AGPU\u3002\u7136\u540E\uFF0C\u6211\u4EEC\u518D\u4ECE\u5934\u5230\u5C3E\u6328\u4E2A\u70B9\u51FB\u6BCF\u4E2A\u4EE3\u7801\u5757\u7684\u8FD0\u884C\u6807\u5FD7\u3002\u53EF\u80FD\u9700\u8981\u7B49\u5F85\u4E00\u5B9A\u7684\u65F6\u95F4\u3002\u5F53\u6211\u4EEC\u8FDB\u884C\u5230\u8BED\u97F3\u5408\u6210\u90E8\u5206\u65F6\uFF0C\u5C31\u53EF\u4EE5\u66F4\u6539\u8981\u8BF4\u7684\u6587\u672C\uFF0C\u5E76\u8BBE\u7F6E\u4FDD\u5B58\u7684\u6587\u4EF6\u540D\u5566\u3002" #@param {type: 'string'}
#text=r'''我要做一个会撒糖糖的小孩子，走一路，糖撒一路，让和我一起走在路上的你甜到心里。'''
##f=open(r"C:\Users\HP\Desktop\语音文字.txt",encoding='utf-8')
##ff=f.read()
##f.close
def getff():
    ff=pyperclip.paste()

    text=ff[0:33]
    f=ff.replace('\n','')
    f=f.replace(' ','')
    f=f.replace('"','')
    f=f.replace(':','：')
    f=f.replace('(','')
    f=f.replace(')','')
    f=f.replace('（','')
    f=f.replace('）','')
    text=f[0:]
    return text
#text='这个模型训练的可以，像不像派蒙声音，我是没玩过原神'


def one(text):
    length_scale = 1.5 #@param {type:"slider", min:0.1, max:3, step:0.05}
    filename = '{}-{}'.format(text[0:15],time.strftime('%Y-%m-%d-%H-%M-%S-%A')) #@param {type: "string"}
    audio_path = r'C:\Users\HP\Downloads\VITS-Paimon\content\VITS-Paimon\语音\{}.wav'.format(filename)
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
    #ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))#only in notebook
    #print(hps.data.sampling_rate)
    sf.write(audio_path,audio,samplerate=hps.data.sampling_rate)
    playsound(audio_path)
    

##def on_click(x,y,button,pressed):
##        control=keyboard.Controller()
##        #time.sleep(0.1)
##        with control.pressed(keyboard.Key.ctrl):
##            control.press('c')
##            control.release('c')
##            text=getff()
##            one(text)
##        print('dd')
def copy_result():
    recent_txt=''
    while True:
            time.sleep(0.2)
            txt=pyperclip.paste()
            if txt!=recent_txt:
                recent_txt=txt
                text=getff()
                one(text)             
def main():
    copy_result()
if __name__=='__main__':
    main()

##if len(text)<444:
##    one(text)
##else:
##    for i in range(len(text)//444+1):
##        one(text[444*i:444*(i+1)])
