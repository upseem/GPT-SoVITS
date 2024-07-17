from pydantic import BaseModel, validator
from typing import List,Literal


class LoraRequest(BaseModel):
    uid: str                # 用户id
    voices: List[str]
    epoch_g: int = 10       # 训练轮数
    epoch_s: int = 25
    lang : str = "en"       # 训练语言 en zh

    @validator('voices', pre=True, always=True)
    def check_image_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError('voices 列表不能为空')
        return v
    
    @validator('uid', pre=True, always=True)
    def check_uuid(cls, v):
        if not v or v == "0" or v == 0:
            raise ValueError('uid 不能为空')
        return v


# 公共返回
class SuccessResponse(BaseModel):
    code: Literal[0] = 0
    msg: str = "处理成功"
    data: dict = {}

# 公共返回
class FailResponse(BaseModel):
    code: Literal[1] = 1
    msg: str = "处理失败"
    data: dict = {}