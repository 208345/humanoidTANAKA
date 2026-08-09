# 開発環境のセットアップ（Windows / PowerShell）
#
#   .\scripts\setup_windows.ps1
#
# 実行がブロックされる場合は先に以下を1回だけ実行:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

Write-Host "== Python の確認 ==" -ForegroundColor Cyan
python --version

Write-Host "`n== 仮想環境の作成 ==" -ForegroundColor Cyan
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
} else {
    Write-Host ".venv は既に存在します"
}

Write-Host "`n== 仮想環境を有効化 ==" -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "`n== 依存関係のインストール ==" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -e ".[dev]"

Write-Host "`n== 動作確認 ==" -ForegroundColor Cyan
python -m control.run --backend dummy --controller zero --steps 100
pytest -q

Write-Host "`nセットアップ完了" -ForegroundColor Green
Write-Host "次回からは .\.venv\Scripts\Activate.ps1 で仮想環境に入れます"
