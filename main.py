from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import uvicorn

from config import config
from llm_service import llm_service, DecisionInput, DecisionResult

app = FastAPI(
    title=config.app.title,
    description=config.app.description,
    version=config.app.version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.app.cors.allow_origins,
    allow_credentials=config.app.cors.allow_credentials,
    allow_methods=config.app.cors.allow_methods,
    allow_headers=config.app.cors.allow_headers,
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

class TextRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    """提供前端页面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["system"])
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": config.app.title,
        "version": config.app.version,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model
    }


@app.post("/api/parse-and-decide", response_model=DecisionResult, tags=["决策"])
async def parse_and_decide(request: TextRequest):
    """解析文本中的选项和条件，并做出决策"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="输入文本不能为空哦~")

    try:
        options = llm_service.parse_options(request.text)

        result = llm_service.make_decision(options)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


# 启动服务
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level
    )
