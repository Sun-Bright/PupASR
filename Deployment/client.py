# client.py
import asyncio
import websockets
import pyaudio

async def record_and_send(uri):
    async with websockets.connect(uri) as ws:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
        try:
            while True:
                audio_data = stream.read(1024)  # 读取 1024 字节的音频数据
                await ws.send(audio_data)
                await asyncio.sleep(0.01)  # 控制发送频率
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    uri = "ws://<服务器IP>:8000/ws/stream"
    asyncio.run(record_and_send(uri))