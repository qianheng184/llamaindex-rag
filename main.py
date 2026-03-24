from engine import RAGEngine
import sys

def main():
    try:
        # 初始化引擎
        rag = RAGEngine()
        print("\n" + "="*30)
        print("RAG 系统已就绪，请输入您的问题")
        print("="*30)
        
        while True:
            question = input("\n问：")
            if question.lower() in ['q', 'exit', 'quit', '退出']:
                break
            
            if not question.strip():
                continue

            print("答：", end="", flush=True)
            response = rag.query(question)
            
            # 流式打印效果
            for token in response.response_gen:
                print(token, end="", flush=True)
            print("\n")

    except Exception as e:
        print(f"\n[程序运行出错]: {e}")

if __name__ == "__main__":
    main()