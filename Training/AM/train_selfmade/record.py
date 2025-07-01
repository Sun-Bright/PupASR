# record_audio_windows_final.py
# 完整版本：支持中断继錄，自动跳过已存文件，记录日志

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import msvcrt  # Windows 检测键盘

# === 用户参数 ===
SPEAKER = "user_5"  # 请根据用户分配
SAVE_DIR = "recordings"  # 存放目录
TEXT_FILE = "txt/recording_plan_user_5.txt"  # 文本路径
LOG_FILE = "recording_log.txt"
SAMPLE_RATE = 16000
CHANNELS = 1

def record_until_enter():
    print("\U0001f399️ 开始录音，说完后按 Enter 结束...")
    recording = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    stream.start()
    try:
        while True:
            frames, _ = stream.read(1024)
            recording.append(frames)
            if msvcrt.kbhit() and msvcrt.getch() == b'\r':  # Enter
                break
    finally:
        stream.stop()
        stream.close()
    return np.concatenate(recording, axis=0)

def main():
    os.makedirs(os.path.join(SAVE_DIR, SPEAKER), exist_ok=True)
    log_fp = open(LOG_FILE, "a", encoding="utf-8")

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, text in enumerate(lines):
        wav_name = f"{idx+1:04d}.wav"
        wav_path = os.path.join(SAVE_DIR, SPEAKER, wav_name)

        if os.path.exists(wav_path):
            print(f"⏭️ 第 {idx+1} 句已存在，跳过: {wav_name}")
            continue

        input(f"\n📢 第 {idx+1} 句：{text}\n▶️ 按 Enter 开始")
        audio = record_until_enter()
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"✅ 已保存：{wav_path}")

        log_fp.write(f"{SPEAKER}/{wav_name}\t{text}\n")
        log_fp.flush()
        time.sleep(0.3)

    log_fp.close()
    print("\n🎉 所有录音完成！")

if __name__ == "__main__":
    main()