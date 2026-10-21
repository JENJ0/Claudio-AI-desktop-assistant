# Builds the venv of Jack's voice worker (Qwen3-TTS + Applio RVC) in models\voices\jack\worker\env.
# Needs uv and Applio 3.7.0 in models\voices\jack\rvc\Applio (the trained voice is not in Git).
# Log: models\voices\jack\worker\setup.log
$models = Join-Path (Split-Path $PSScriptRoot -Parent) "models"
$jack = Join-Path $models "voices\jack"
$worker = Join-Path $jack "worker"
$applio = Join-Path $jack "rvc\Applio"
if (-not (Test-Path (Join-Path $applio "requirements-engine.txt"))) {
    Write-Error "Applio not found in $applio"
    exit 1
}
New-Item -ItemType Directory -Force $worker | Out-Null
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$log = Join-Path $worker "setup.log"
# Applio pins transformers 5.13.1, faster-qwen3-tts 0.5.4 needs 5.15.1: the override wins.
"transformers==5.15.1" | Out-File (Join-Path $worker "override.txt") -Encoding ascii
$python = Join-Path $worker "env\Scripts\python.exe"
uv venv --python 3.12 (Join-Path $worker "env") *> $log
uv pip install --python $python -r (Join-Path $applio "requirements-engine.txt") "faster-qwen3-tts==0.5.4" soundfile `
    --override (Join-Path $worker "override.txt") `
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match *>> $log
& $python -c "import torch, transformers, faster_qwen3_tts; print('OK', torch.__version__, transformers.__version__, torch.cuda.is_available())"
