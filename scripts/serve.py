#!/usr/bin/env python3
"""书的服务器：静态文件 + 批注自动收集 API。

替代 `python -m http.server`，用法：
    python scripts/serve.py

功能：
1. 在 http://localhost:8123 提供书的静态文件（同 http.server）
2. 接收浏览器端批注工具的自动 POST，实时写入 reviews/ 目录

批注工具（review-header.html）在 localhost 环境下会自动
把每条批注 POST 到 /api/annotations，无需手动导出。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / "_book"
REVIEWS_DIR = ROOT / "reviews"


class Annotation:
    """单个批注的存储动作。"""
    counter = 0


class BookServer(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BOOK_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/api/annotations":
            self.handle_annotation()
        elif self.path == "/api/sync":
            self.handle_sync()
        else:
            self.send_error(404)

    def handle_annotation(self):
        """接收单条批注，追加到 reviews/auto-<chapter>.json。"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self.send_json(400, {"ok": False, "error": "invalid JSON"})
            return

        chapter = body.get("chapter", "unknown")
        REVIEWS_DIR.mkdir(exist_ok=True)

        # 追加模式：每章一个文件，按 id 去重
        filepath = REVIEWS_DIR / f"auto-{chapter}.json"
        if filepath.exists():
            try:
                existing = json.loads(filepath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {"chapter": chapter, "comments": []}
        else:
            existing = {"chapter": chapter, "comments": []}

        # 按 id 去重（更新或追加）
        comments = existing.get("comments", [])
        comment_id = body.get("id")
        found = False
        for i, c in enumerate(comments):
            if c.get("id") == comment_id:
                comments[i] = body
                found = True
                break
        if not found:
            comments.append(body)

        existing["comments"] = comments
        existing["lastUpdated"] = datetime.now().isoformat()

        filepath.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        self.send_json(200, {"ok": True, "total": len(comments)})

    def handle_sync(self):
        """接收整章批注批量同步。"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self.send_json(400, {"ok": False, "error": "invalid JSON"})
            return

        chapter = body.get("chapter", "unknown")
        REVIEWS_DIR.mkdir(exist_ok=True)
        filepath = REVIEWS_DIR / f"auto-{chapter}.json"
        filepath.write_text(
            json.dumps(body, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        n = len(body.get("comments", []))
        self.send_json(200, {"ok": True, "total": n})

    def send_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """静默静态文件请求，只显示 API 调用。"""
        if "/api/" in str(args[0] if args else ""):
            super().log_message(format, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123

    # 尝试绑定端口，被占则自动换
    for try_port in range(port, port + 10):
        try:
            server = HTTPServer(("localhost", try_port), BookServer)
            break
        except PermissionError:
            print(f"端口 {try_port} 被占用，尝试 {try_port + 1}...")
            continue
    else:
        print("错误：8123-8132 都被占用，请用 python scripts/serve.py <端口号> 指定")
        sys.exit(1)

    if try_port != port:
        print(f"（原端口 {port} 被占，已自动切换到 {try_port}）")
    print(f"书已启动: http://localhost:{try_port}/")
    print(f"批注自动保存到: {REVIEWS_DIR}/")
    print(f"按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()
