import yaml
from pathlib import Path
from typing import Optional, List  # 导入List类型
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"


class LLMParameters(BaseModel):
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 30


class LLMConfig(BaseModel):
    provider: str = "openai-compatible"
    api_base: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    parameters: LLMParameters = LLMParameters()


class CORSConfig(BaseModel):
    # 将list[str]改为List[str]
    allow_origins: List[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: List[str] = ["*"]
    allow_headers: List[str] = ["*"]


class AppConfig(BaseModel):
    title: str = "决策助手API"
    description: str = "一个有趣的决策助手API"
    version: str = "1.0.0"
    cors: CORSConfig = CORSConfig()


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    llm: LLMConfig = LLMConfig()
    app: AppConfig = AppConfig()


def load_config(config_path: str) -> Config:
    """加载YAML配置文件"""
    try:
        path = Path(config_path)
        if not path.exists():
            print(f"配置文件 {config_path} 不存在，将使用默认配置")
            return Config()

        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return Config(**config_data)

    except Exception as e:
        print(f"加载配置文件出错: {str(e)}，将使用默认配置")
        return Config()


# 加载配置单例
config = load_config("config.yaml")
