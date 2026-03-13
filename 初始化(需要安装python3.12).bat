@echo off
chcp 65001 >nul
cd /d %~dp0

echo ========================================
echo   RAG MCP 知识库系统 - 环境初始化
echo ========================================
echo.

echo [1/5] 配置 pip 阿里源...
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
echo.

echo [2/5] 创建虚拟环境...
if exist venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    echo 虚拟环境创建成功
)
echo.

echo [3/5] 激活虚拟环境...
call venv\Scripts\activate.bat
echo.

echo [4/5] 安装 PyTorch (CUDA 12.8)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo.

echo [5/5] 安装其他依赖...
pip install -r requirements.txt
echo.

echo ========================================
echo   初始化完成！
echo ========================================
echo.
echo 后续操作：
echo   - 启动后端服务：双击 运行后端.bat
echo   - 启动GUI界面：双击 运行GUI.bat
echo.
pause
