#!/usr/bin/env python3
"""RAG System v2.0 - 在线查询 CLI 客户端"""
import os
import sys
import httpx
from typing import Optional

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class RAGClient:
    """RAG API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=60.0)
    
    def query(self, question: str, top_k: int = 5, use_cache: bool = True) -> dict:
        """发送查询请求"""
        url = f"{self.base_url}/query"
        payload = {
            "query": question,
            "top_k": top_k,
            "use_cache": use_cache
        }
        
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def stream_query(self, question: str, top_k: int = 5):
        """流式查询"""
        url = f"{self.base_url}/query/stream"
        payload = {
            "query": question,
            "top_k": top_k
        }
        
        with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line.strip():
                    # 解析 SSE 数据
                    if line.startswith("data: "):
                        data = line[6:]  # 移除 "data: "
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            import json
                            chunk_data = json.loads(data)
                            
                            if 'content' in chunk_data:
                                yield chunk_data['content']
                            elif 'done' in chunk_data and chunk_data['done']:
                                break
                            elif 'error' in chunk_data:
                                yield f"\n[错误：{chunk_data['error']}]"
                                break
                        except json.JSONDecodeError:
                            continue
    
    def health_check(self) -> dict:
        """健康检查"""
        url = f"{self.base_url}/health"
        response = self.client.get(url)
        return response.json()
    
    def close(self):
        """关闭客户端"""
        self.client.close()


def print_streaming_response(client: RAGClient, question: str, top_k: int = 5):
    """打印流式响应"""
    print("\n正在生成回答...\n")
    
    full_answer = ""
    try:
        for chunk in client.stream_query(question, top_k=top_k):
            print(chunk, end="", flush=True)
            full_answer += chunk
        
        print("\n")
        
    except Exception as e:
        print(f"\n\n✗ 流式查询失败：{e}")


def interactive_mode(base_url: str):
    """交互式问答模式"""
    print("=" * 60)
    print("RAG System v2.0 - 交互式问答")
    print("=" * 60)
    
    client = RAGClient(base_url=base_url)
    
    try:
        # 健康检查
        print("\n正在检查服务状态...")
        health = client.health_check()
        
        if health.get('available', False):
            print(f"✓ 服务已连接：{health.get('status', 'unknown')}")
        else:
            print(f"⚠ 服务状态异常：{health.get('status', 'unknown')}")
        
        print("\n" + "=" * 60)
        print("请输入您的问题（输入 'q' 退出，输入 's' 切换流式模式）")
        print("=" * 60)
        
        stream_mode = True  # 默认使用流式模式
        
        while True:
            try:
                question = input(f"\n{'[流式]' if stream_mode else '[普通]'} 问：").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['q', 'quit', 'exit', '退出']:
                    print("\n再见！")
                    break
                
                if question.lower() in ['s', 'stream', 'streaming']:
                    stream_mode = not stream_mode
                    print(f"已切换到{'流式' if stream_mode else '普通'}模式")
                    continue
                
                # 执行查询
                if stream_mode:
                    print_streaming_response(client, question)
                else:
                    response = client.query(question)
                    print(f"\n答：{response['answer']}")
                    
                    # 显示来源信息
                    if response.get('contexts'):
                        print(f"\n参考了 {len(response['contexts'])} 个相关片段")
                    
            except KeyboardInterrupt:
                print("\n\n操作已中断")
                continue
            except Exception as e:
                print(f"\n✗ 查询失败：{e}")
    
    finally:
        client.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System CLI 客户端")
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:8000",
        help="API 服务地址（默认：http://localhost:8000）"
    )
    parser.add_argument(
        "--question", "-q",
        help="直接提问（不进入交互模式）"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出"
    )
    
    args = parser.parse_args()
    
    # 如果有直接的问题，执行单次查询
    if args.question:
        client = RAGClient(base_url=args.url)
        
        try:
            if not args.no_stream:
                print_streaming_response(client, args.question)
            else:
                response = client.query(args.question)
                print(f"\n答：{response['answer']}")
        finally:
            client.close()
    else:
        # 进入交互模式
        interactive_mode(args.url)


if __name__ == "__main__":
    main()
