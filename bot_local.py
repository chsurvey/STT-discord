import discord
from discord.ext import commands
import os
import asyncio
import multiprocessing
import time
import logging
from dotenv import load_dotenv
from pydub import AudioSegment
import io
import audioop

# ---------------------------------------------------------
# [별도 프로세스] AI 추론 워커 (STT + 번역)
# ---------------------------------------------------------
def inference_worker(input_queue, output_queue, model_config):
    """
    무거운 AI 모델들을 로드하고 추론을 담당하는 독립 프로세스입니다.
    봇의 메인 루프(Event Loop)를 차단하지 않기 위해 별도로 돕니다.
    """
    print(f"🔄 [Worker] AI 모델 로딩 중... (PID: {os.getpid()})")
    
    try:
        # 1. 최적화된 STT 로드 (Faster-Whisper + INT8 양자화)
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(
            model_config['stt_model_size'], 
            device="cuda",  # GPU 사용 (없으면 cpu)
            compute_type="int8" # 양자화 적용 (속도 ↑, 메모리 ↓)
        )

        # 2. 최적화된 LLM 로드 (Llama.cpp + GGUF 4bit)
        # 번역 전용 프롬프트를 위해 시스템 메시지 설정이 가능한 모델 권장 (예: Qwen, Gemma)
        from llama_cpp import Llama
        llm_model = Llama(
            model_path=model_config['llm_model_path'],
            n_gpu_layers=-1, # 가능한 모든 레이어를 GPU로
            n_ctx=512,       # 번역이므로 컨텍스트는 짧게
            verbose=False
        )
        
        print("✅ [Worker] 모델 로딩 완료. 대기 중...")

    except Exception as e:
        print(f"❌ [Worker] 모델 로딩 실패: {e}")
        return

    while True:
        try:
            # 큐에서 작업 가져오기 (audio_bytes, user_id, target_lang)
            task = input_queue.get()
            if task is None: break # 종료 신호

            user_id, audio_bytes, target_lang = task
            start_time = time.time()

            # --- STT 추론 ---
            # Bytes -> Float32 Array 변환 (faster-whisper용)
            audio_segment = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=48000, channels=2)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            # pydub 객체에서 raw data 추출 후 numpy 변환 (생략하고 파일처럼 전달 가능)
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            segments, _ = stt_model.transcribe(wav_io, beam_size=5, language=None) # 언어 자동 감지
            original_text = " ".join([s.text for s in segments]).strip()

            if not original_text:
                continue

            # --- LLM 번역 (프롬프트 엔지니어링) ---
            # 한국어로 번역 요청
            prompt = f"""<|im_start|>system
You are a professional translator. Translate the following text into natural Korean.<|im_end|>
<|im_start|>user
{original_text}<|im_end|>
<|im_start|>assistant
"""
            output = llm_model(
                prompt, 
                max_tokens=128, 
                stop=["<|im_end|>", "\n"], 
                temperature=0.3
            )
            translated_text = output['choices'][0]['text'].strip()
            
            inference_time = time.time() - start_time
            
            # 결과 전송
            output_queue.put({
                "user_id": user_id,
                "original": original_text,
                "translated": translated_text,
                "time": inference_time
            })

        except Exception as e:
            print(f"⚠️ [Worker] 추론 에러: {e}")


# ---------------------------------------------------------
# [메인 프로세스] 디스코드 봇
# ---------------------------------------------------------
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# 설정
MODEL_CONFIG = {
    "stt_model_size": "medium",  # tiny, base, small, medium, large-v3
    # 다운로드 받은 GGUF 파일 경로 (예: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)
    "llm_model_path": "./models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" 
}

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 프로세스 간 통신을 위한 큐
task_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()

class LocalVADSink(discord.sinks.Sink):
    def __init__(self, task_queue, filters=None):
        if filters is None: filters = discord.sinks.default_filters
        super().__init__(filters=filters)
        self.task_queue = task_queue
        self.user_data = {}
        
        # VAD 파라미터
        self.SILENCE_THRESHOLD = 1000
        self.SILENCE_LIMIT = 0.5

    def get_user_data(self, user):
        if user not in self.user_data:
            self.user_data[user] = {
                "buffer": bytearray(),
                "silence_start": None,
                "is_speaking": False
            }
        return self.user_data[user]

    @discord.sinks.Filters.container
    def write(self, data, user):
        ud = self.get_user_data(user)
        try: rms = audioop.rms(data, 2)
        except: rms = 0

        # VAD Logic
        if rms > self.SILENCE_THRESHOLD:
            ud["silence_start"] = None
            ud["is_speaking"] = True
        else:
            if ud["silence_start"] is None:
                ud["silence_start"] = time.time()

        ud["buffer"] += data
        now = time.time()

        # 침묵 감지 시 버퍼 처리
        if (ud["is_speaking"] and 
            ud["silence_start"] is not None and 
            (now - ud["silence_start"]) > self.SILENCE_LIMIT):
            
            # 너무 짧은 오디오 무시 (노이즈 필터링)
            if len(ud["buffer"]) > 30000: 
                # 큐에 작업 등록 (Non-blocking)
                audio_copy = bytes(ud["buffer"])
                self.task_queue.put((user, audio_copy, "ko"))
                # print(f"📥 [Main] 오디오 큐 전송 완료 (User: {user})")

            ud["buffer"] = bytearray()
            ud["is_speaking"] = False
            ud["silence_start"] = None

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    # 백그라운드 태스크: 결과 큐 모니터링
    bot.loop.create_task(check_results())

async def check_results():
    """결과 큐를 주기적으로 확인하여 디스코드에 메시지 전송"""
    while True:
        try:
            # Non-blocking 방식으로 큐 확인
            while not result_queue.empty():
                result = result_queue.get_nowait()
                user_id = result["user_id"]
                original = result["original"]
                translated = result["translated"]
                infer_time = result["time"]

                # 메시지를 보낼 채널 찾기 (간소화를 위해 음성 채널이 있는 서버의 첫 번째 텍스트 채널 등 로직 필요)
                # 여기서는 예시로 가장 최근에 명령어를 친 채널 등을 저장해서 써야 함.
                # 편의상 'join' 명령어를 친 컨텍스트의 채널을 전역으로 쓴다고 가정하거나
                # user_id로 DM을 보내거나 할 수 있습니다. 
                
                # 예시: 글로벌 변수나 딕셔너리에 저장된 active_channel 사용
                if active_channel:
                    await active_channel.send(
                        f"⚡ **{translated}**\n"
                        f"└ `({original})` [⏱️ {infer_time:.2f}s]"
                    )
            
            await asyncio.sleep(0.1) # CPU 과부하 방지
        except Exception as e:
            print(f"Result loop error: {e}")
            await asyncio.sleep(1)

active_channel = None

@bot.command("join")
async def join(ctx):
    global active_channel
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        active_channel = ctx.channel
        
        # 싱크 시작
        ctx.voice_client.start_recording(
            LocalVADSink(task_queue),
            finished_callback,
            ctx.channel
        )
        await ctx.send(f"✅ **로컬 AI 통역 시작** (Model: Faster-Whisper + GGUF)")
    else:
        await ctx.send("음성 채널에 먼저 들어가주세요.")

async def finished_callback(sink, channel, *args):
    await channel.send("세션 종료.")

@bot.command("leave")
async def leave(ctx):
    if ctx.voice_client:
        ctx.voice_client.stop_recording()
        await ctx.voice_client.disconnect()

if __name__ == "__main__":
    # 윈도우/리눅스 멀티프로세싱 호환성
    multiprocessing.freeze_support()
    
    # 1. AI 워커 프로세스 시작
    worker = multiprocessing.Process(
        target=inference_worker, 
        args=(task_queue, result_queue, MODEL_CONFIG)
    )
    worker.daemon = True # 메인 프로세스 종료 시 같이 종료
    worker.start()

    # 2. 봇 실행
    bot.run(DISCORD_BOT_TOKEN)