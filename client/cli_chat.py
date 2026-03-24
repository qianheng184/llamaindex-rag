#!/usr/bin/env python3
"""CLI 聊天客户端

通过 HTTP 调用 RAG API 服务，提供命令行交互界面

使用方法:
    python client/cli_chat.py
"""
import sys
import httpx


def chat_client():
    """CLI 聊天客户端主函数"""
    api_url = "http://localhost:8000/query"
    
    print("=" * 60)
    print("RAG 智能助手 - CLI 客户端")
    print("=" * 60)
    print("请输入您的问题，输入 'q'、'quit'、'exit' 或 '退出' 结束对话")
    print("=" * 60)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n问：").strip()
            
            # 检查退出命令
            if question.lower() in ['q', 'quit', 'exit', '退出']:
                print("\n感谢使用，再见!")
                break
            
            # 跳过空输入
            if not question:
                continue
            
            # 调用 API
            print("答：", end="", flush=True)
            
            try:
                response = httpx.post(
                    api_url,
                    json={"query": question},
                    timeout=60.0
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('answer', '')
                    print(answer)
                else:
                    print(f"\n[错误] API 返回错误：{response.status_code}")
                    print(f"详情：{response.text}")
            
            except httpx.ConnectError as e:
                print(f"\n[连接错误] 无法连接到 API 服务：{e}")
                print("请确保 API 服务正在运行：uvicorn api.app:app --reload")
            except httpx.TimeoutException as e:
                print(f"\n[超时错误] 请求超时：{e}")
            except Exception as e:
                print(f"\n[请求错误] {e}")
        
        except KeyboardInterrupt:
            print("\n\n检测到中断，退出程序...")
            break
        except EOFError:
            print("\n\n检测到输入结束，退出程序...")
            break


if __name__ == "__main__":
    chat_client()
