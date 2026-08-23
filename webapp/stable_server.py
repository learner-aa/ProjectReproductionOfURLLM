#!/usr/bin/env python3
"""
稳定的前端服务 - 内置自动重启

用法:
    setsid /root/miniconda3/envs/urllm/bin/python stable_server.py &
    disown

特点:
- 进程崩溃自动重启
- 端口占用自动处理
- 完全脱离终端 (setsid)
"""
import http.server
import socketserver
import os
import sys
import time
import signal

PORT = 6006
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
LOG_FILE = "/tmp/stable_server.log"


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if "." not in os.path.basename(self.path):
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, format, *args):
        # 简化日志，只记录错误
        pass


class ReuseAddrTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        f.flush()


def run_server():
    """运行一次HTTP服务，直到出错返回"""
    try:
        with ReuseAddrTCPServer(("0.0.0.0", PORT), NoCacheHTTPRequestHandler) as httpd:
            log(f"服务启动: http://0.0.0.0:{PORT} -> {DIRECTORY}")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            log(f"端口 {PORT} 被占用，等待5秒后重试...")
            time.sleep(5)
        else:
            log(f"OSError: {e}")
            time.sleep(3)
    except Exception as e:
        log(f"异常: {e}")
        time.sleep(3)


def main():
    # 脱离终端
    try:
        os.setsid()
    except OSError:
        pass

    # 忽略终端信号
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    log(f"=== 守护进程启动 PID={os.getpid()} ===")

    # 主循环：崩溃自动重启
    while True:
        run_server()


if __name__ == "__main__":
    main()
