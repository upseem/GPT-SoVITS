from server import *
from my_file import *
from my_model import *
from my_util import *

class MyTrainer:
    def __init__(self,req: LoraRequest):
        self.my_lang    = req.lang     # en zh
        self.R_PATH     = R_PATH
        self.V_PATH     = V_PATH
        self.S_PATH     = S_PATH
        self.A_PATH     = A_PATH
        self.A_LIST     = A_LIST
        self.epoch_g    = req.epoch_g
        self.epoch_s    = req.epoch_s
        if self.my_lang == "zh":
            self.my_asr_model        = "达摩 ASR (中文)"
            self.my_asr_model_size   = "large"
            self.my_asr_lang         = "zh"
        else:
            self.my_asr_model        = "Faster Whisper (多语种)"
            self.my_asr_model_size   = "large-v3"
            self.my_asr_lang         = "auto"

        self.bert_pretrained_dir     = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large" 	    #--预训练的 BERT 模型路径
        self.ssl_pretrained_dir      = "GPT_SoVITS/pretrained_models/chinese-hubert-base" 				    #--预训练的 SSL 模型路径
        self.pretrained_s2G_path     = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        self.pretrained_s2D_path     = "GPT_SoVITS/pretrained_models/s2D488k.pth"
        self.pretrained_s1           = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    
    # 切音频
    def slice_voice(self):
        inp         = self.V_PATH   #--slice_inp_path 音频自动切分输入路径，可文件可文件夹
        opt_root    = self.S_PATH   #--slice_opt_root 切分后的子音频的输出根目录
        threshold   = -34           #--音量小于这个值视作静音的备选切割点
        min_length  = 4000          #--每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval= 300           #--最短切割间隔
        hop_size    = 10            #--怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept= 500           #--切完后静音最多留多长
        _max        = 0.9           #--归一化后最大值多少
        alpha       = 0.25          #--混多少比例归一化后音频进来
        n_parts     = 4             #--切割使用的进程数 默认4

        result_generator = open_slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_parts)
        
        # 使用 next() 函数获取生成器的输出
        try:
            while True:
                result = next(result_generator)
                print(result)
        except StopIteration:
            print("切音频,生成器执行完成")
            
    # 转中文
    def to_open_asr(self):
        asr_inp_dir     = self.S_PATH               #--输入 -i
        asr_opt_dir     = self.A_PATH 		        #--输出 -o
        asr_model       = self.my_asr_model         # asr模型
        asr_model_size  = self.my_asr_model_size    #--模型类别 -s
        asr_lang        = self.my_asr_lang			#--模型语言 -l 
        result_generator = open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang)
        
        # 使用 next() 函数获取生成器的输出
        try:
            while True:
                result = next(result_generator)
                print(result)
        except StopIteration:
            print("转中文,生成器执行完成")

    #一键三连
    def to_open1abc(self):
        inp_text        = self.A_LIST    #--*文本标注文件
        inp_wav_dir     = self.S_PATH  # 														--音频文件目录
        gpu_numbers1a   = "0-0"
        gpu_numbers1Ba  = "0-0"
        gpu_numbers1c   = "0-0"
        result_generator = open1abc(inp_text,inp_wav_dir,self.exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,self.bert_pretrained_dir,self.ssl_pretrained_dir,self.pretrained_s2G_path)
        try:
            while True:
                result = next(result_generator)
                print(result)
        except StopIteration:
            print("生成器执行完成")

    # S模型
    def s_w_open1Ba(self):
        batch_size              = 12   			#--每张显卡的batch_size 		minimum=1,maximum=40,step=1
        total_epoch             = self.epoch_s 	#--总训练轮数total_epoch		minimum=1,maximum=25,step=1
        # exp_name =  "xxx"					    #--*实验/模型名
        text_low_lr_rate        = 0.4 			#--文本模块学习率权重 			  minimum=0.2,maximum=0.6,step=0.05
        if_save_latest          = True			#--是否仅保存最新的ckpt文件以节省硬盘空间
        if_save_every_weights   = True          #--是否在每次保存时间点将最终小模型保存至weights文件夹
        save_every_epoch        = total_epoch	#--保存频率save_every_epoch minimum=1,maximum=25,step=1
        gpu_numbers1Ba          = gpus          #-- GPU卡号以-分割，每个卡号一个进程

        result_generator = open1Ba(batch_size,total_epoch,self.exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,self.pretrained_s2G_path,self.pretrained_s2D_path)
        try:
            while True:
                result = next(result_generator)
                print(result)
        except StopIteration:
            print("生成器执行完成")
            
    # G模型
    def g_w_open1Bb(self):
        batch_size              = 6   			#--每张显卡的batch_size 		minimum=1,maximum=40,step=1
        total_epoch             = self.epoch_g 	#--总训练轮数total_epoch		minimum=2,maximum=50,step=1
        # exp_name="xxx"						#--*实验/模型名
        if_dpo                  = True			#--是否开启dpo训练选项(实验性) 			
        if_save_latest          = True			#--是否仅保存最新的ckpt文件以节省硬盘空间
        if_save_every_weights   = True          #--是否在每次保存时间点将最终小模型保存至weights文件夹
        save_every_epoch        = total_epoch	#--保存频率save_every_epoch minimum=1,maximum=50,step=1
        gpu_numbers             = gpus          #-- GPU卡号以-分割，每个卡号一个进程
        

        result_generator = open1Bb(batch_size,total_epoch,self.exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,self.pretrained_s1)
        
        try:
            while True:
                result = next(result_generator)
                print(result)
        except StopIteration:
            print("生成器执行完成")

    def train_voice(self):
        print("---------1.切片开始---------")        
        self.slice_voice()
        print("---------2.文字开始---------")   
        self.to_open_asr()
        print("---------3.三连开始---------")  
        self.to_open1abc()
        print("---------3.s_w开始---------")  
        self.s_w_open1Ba()
        print("---------4.g_w开始---------")  
        self.g_w_open1Bb()
        print("-----------结束------------") 
