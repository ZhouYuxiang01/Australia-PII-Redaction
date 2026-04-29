# Quickstart — 启动 / 关闭

> 整个系统两段:**远端服务器**跑 backend(模型 + FastAPI),**Windows 本机**用浏览器看前端。中间靠 SSH 隧道把端口转过来。

---

## 一次性准备

确认远端服务器(GB10 / `100.91.98.45`)上有这个目录:

```
/home/admin/ZYX/redaction-wrapper/
```

---

## 启动(每次开机后做一次)

### 步骤 1 — 远端启动 backend

在 SSH 进远端后,在 `(base) admin@aitopatom-5c4b:~$` 提示符下执行:

```bash
cd /home/admin/ZYX/redaction-wrapper
tmux kill-session -t redaction_wrapper 2>/dev/null
tmux new-session -d -s redaction_wrapper 'WRAPPER_PORT=8091 ./scripts/run_server.sh'

# 等 5 秒,确认起来了
sleep 5
curl -s http://127.0.0.1:8091/api/health | head
```

看到 `{"status":"ok","backend":...}` 就成功了。
看到 `Connection refused` 就再等几秒(模型加载第一次会慢),或者 `tmux attach -t redaction_wrapper` 看启动日志。

### 步骤 2 — Windows 本机起 SSH 隧道

打开 PowerShell,**复制下面整行**(注意是一行):

```powershell
ssh -i C:\Users\zyx62\Desktop\5703test\.ssh\modernbert_distill_ed25519 -L 8091:127.0.0.1:8091 admin@100.91.98.45
```

回车后会进入远端 shell,**这个窗口不要关**(关了隧道就断)。

### 步骤 3 — 浏览器打开前端

直接访问:

```
http://127.0.0.1:8091/
```

交互式 API 文档:

```
http://127.0.0.1:8091/docs
```

---

## 关闭

### 关前端访问(只断 Windows 这边的隧道)

回到 Windows 那个 PowerShell 窗口,**Ctrl + D** 或者 `exit` 退出 SSH。

### 关 backend(远端)

在远端 shell:

```bash
tmux kill-session -t redaction_wrapper
```

确认关掉了:

```bash
curl -s http://127.0.0.1:8091/api/health
# 应该返回 connection refused
```

### 一键全停(远端)

```bash
tmux kill-session -t redaction_wrapper 2>/dev/null
fuser -k 8091/tcp 2>/dev/null   # 兜底:杀掉所有占 8091 的进程
```

---

## 切换模型(可选)

默认跑 OPF v3。如果要换模型,在**步骤 1 第三行**改成:

```bash
# Qwen 9B LoRA
tmux new-session -d -s redaction_wrapper 'WRAPPER_PORT=8091 ./scripts/run_server.sh \
  -b configs/backends/qwen-9b-lora.json \
  -p configs/policies/qwen-9b-lora-default-v1.json'

# Qwen 4B Full SFT
tmux new-session -d -s redaction_wrapper 'WRAPPER_PORT=8091 ./scripts/run_server.sh \
  -b configs/backends/qwen-4b-full.json \
  -p configs/policies/qwen-4b-full-default-v1.json'
```

切换后第一次请求会重新加载模型(OPF ~5 秒,Qwen ~30-60 秒)。

---

## 状态检查 / 查日志

```bash
# 是否在跑(远端)
curl -s http://127.0.0.1:8091/api/health | python3 -m json.tool

# 实时日志
tmux attach -t redaction_wrapper        # Ctrl+B 然后 D 退出但不杀
# 或
tail -f /home/admin/ZYX/redaction-wrapper/scripts/logs/redaction_wrapper_*_8091.log

# 看 tmux session 列表
tmux ls
```

---

## 常见问题

| 现象 | 解决 |
|---|---|
| 浏览器 `connection refused` | 远端 backend 没启动 → 步骤 1 |
| 浏览器一直转圈 | SSH 隧道断了 → 步骤 2 重连 |
| `channel 3: open failed: Connection refused` 一直刷屏 | backend 还没起 → 等几秒,或步骤 1 |
| 端口被占 | `fuser -k 8091/tcp` 后重启 |
| PowerShell 报 `-L: 无法将"-L"项识别为 cmdlet` | 命令换行用了 bash 的 `\` → 改成单行 |
| 改了 config 没生效 | 必须重启 backend(步骤 1) |
| 第一次请求很慢 | 模型加载,OPF ~5s / Qwen ~30-60s,后续是 ms 级 |

---

## 快速命令速查

| 操作 | 命令 |
|---|---|
| **启 backend** | `cd /home/admin/ZYX/redaction-wrapper && tmux new-session -d -s redaction_wrapper 'WRAPPER_PORT=8091 ./scripts/run_server.sh'` |
| **关 backend** | `tmux kill-session -t redaction_wrapper` |
| **重启 backend** | 关 + 启 |
| **看日志** | `tmux attach -t redaction_wrapper` |
| **起 SSH 隧道** | `ssh -i ...路径\modernbert_distill_ed25519 -L 8091:127.0.0.1:8091 admin@100.91.98.45` |
| **断 SSH 隧道** | `exit` 或关掉 PowerShell 窗口 |
| **打开前端** | 浏览器 `http://127.0.0.1:8091/` |
| **健康检查** | `curl -s http://127.0.0.1:8091/api/health` |
