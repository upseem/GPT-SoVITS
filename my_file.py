import os


# 根目录
R_PATH = os.path.dirname(os.path.abspath(__file__))
T_PATH =  os.path.join(R_PATH,"train_data") # 训练目录

T_FOLDERS = [
    "voice",        # 原始音频
    "slicer_opt",   # 切片后的音频
    "asr_opt"       # 处理日志
    "model"
    "logs"
]

V_PATH = os.path.join(T_PATH,"voice")
S_PATH = os.path.join(T_PATH,"slicer_opt")
A_PATH = os.path.join(T_PATH,"asr_opt")
A_LIST = os.path.join(T_PATH,"asr_opt","asr.list")
M_PATH = os.path.join(T_PATH,"model")
L_PATH = os.path.join(T_PATH,"logs")