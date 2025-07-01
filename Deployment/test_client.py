import asyncio
import websockets
import json

async def asr_listener():
    uri = "ws://127.0.0.1:8000/ws/asr"
    async with websockets.connect(uri) as ws:
        print("连接已建立")
        try:
            async for message in ws:
                data = json.loads(message)
                print("实时识别:", data["text"], "置信度:", data["confidence"])
        except:
            print("连接关闭")

asyncio.run(asr_listener())
