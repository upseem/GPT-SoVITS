import os,sys,uuid

from fastapi import FastAPI
from my_model import *
from my_util import *
from my_train import MyTrainer

## 添加到系统目录
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))


print("系统注册目录",sys.path)
app = FastAPI()



@app.post("/train")
async def create_item(req: LoraRequest):
    
    uuid_1 = uuid.uuid4()
    ## 创建uid目录-子目录
    create_t_all_path(req.uid,uuid_1)

    ## 下载音频
    download_voices(req.voices)

    try:
        trainer = MyTrainer(req)
        trainer.train_voice()
    except Exception as e:
        print(f"Error occurred during image handling: {e}")
    finally:
        #训练完删除目录
        # del_uidd_path(req.uid,uuid_1)
        # torch.cuda.empty_cache()
        print("训练脚本已经删除")
        
## 模型转移

## 参考音频挑选
