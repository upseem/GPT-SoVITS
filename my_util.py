import os,shutil,requests
from io import BytesIO

from my_file import *

def create_folders(base_path: str,SUB_LIST:list):
    """在给定的基路径下创建所需的子文件夹。"""
    for folder in SUB_LIST:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

def create_path(file_path: str):
    """在给定的基路径下创建所需的子文件夹。"""
    os.makedirs(file_path, exist_ok=True)


def create_t_all_path(uid: str,uuid_1):
    """在给定的基路径下创建所需的子文件夹。"""
    t_path = os.path.join(T_PATH,uid,uuid_1)
    
    os.makedirs(t_path, exist_ok=True)
    for folder in T_FOLDERS:
        os.makedirs(os.path.join(t_path, folder), exist_ok=True)
    
def download_voices(voices: list, audio_format: str = "mp3"):
    for index, voices_url in enumerate(voices, start=1):
        try:
            # 检查URL是否以指定的前缀开始
            if voices_url.startswith("https://storage.googleapis.com"):
                # 替换URL中的域名
                voices_url = voices_url.replace("https://storage.googleapis.com", "https://google.qisi.co")
            response = requests.get(voices_url, timeout=15)  # 设置超时时间为15秒
            # 检查响应是否正确
            if response.status_code == 200:
                # 使用BytesIO读取下载的内容
                audio_data = BytesIO(response.content)
                # 保存音频到目标文件夹
                save_path = os.path.join(V_PATH, f"{index}.{audio_format}")
                with open(save_path, 'wb') as f:
                    f.write(audio_data.read())
                print(f'下载并保存第{index}个音频到 {save_path}')
            else:
                print(f"无法从 URL 下载音频: {voices_url}")
        except requests.Timeout:
            print(f'第{index}个下载超时')
        except requests.RequestException as e:
            print(f"请求异常: {e}")


#删除目录 images_目录
def del_uidd_path(uid,uuid_1):
    t_path = os.path.join(T_PATH,uid,uuid_1)
    # 检查目录是否存在
    if os.path.exists(t_path):
        # 删除目录及其所有内容
        shutil.rmtree(t_path)
        print(f"Directory '{t_path}' has been deleted.")
    else:
        print(f"Directory '{t_path}' does not exist.")