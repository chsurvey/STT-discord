import discord
from discord.ext import commands
import os
import asyncio
import multiprocessing
import time
import io
import audioop
from dotenv import load_dotenv
from pydub import AudioSegment

# 로그 설정
import logging
logging.getLogger("discord.client").setLevel(logging.INFO)
logging.getLogger("discord.gateway").setLevel(logging.INFO)

# ==========================================
# ⚙️ 설정 (GPU 최적화)
# ==========================================
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

MODEL_CONFIG = {
    # [STT] Whisper 설정 (GPU)
    "stt_model_size": "medium",   # GPU면 'medium'이나 'large-v3'도 충분히 돌립니다!
    "stt_device": "cuda",         # GPU 사용
    "stt_compute_type": "float16",# GPU에서는 float16이 가장 빠름

    # [LLM] GGUF 모델 설정 (GPU)
    # 모델 파일이 models 폴더에 있는지 꼭 확인하세요.
    "llm_model_path": "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "llm_n_gpu_layers": -1,       # -1: 모든 레이어를 GPU VRAM에 올림 (가장 빠름)
}

# 언어 매핑
LANG_MAP = {
    "en": "en", "영어": "en",
    "ja": "ja", "일본어": "ja",
    "zh": "zh", "중국어": "zh",
    "ko": "ko", "한국어": "ko",
    "es": "es", "fr": "fr",
    "auto": None
}

# ==========================================
# 🧠 [Process 2] AI 추론 워커 (GPU 사용)
# ==========================================
def inference_worker(input_queue, output_queue, config):
    print(f"🔄 [Worker] GPU 모드로 모델 로딩 시작... (PID: {os.getpid()})")
    
    try:
        # 1. Faster-Whisper 로드 (GPU 설정)
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(
            config['stt_model_size'], 
            device=config['stt_device'],             # "cuda"
            compute_type=config['stt_compute_type']  # "float16"
        )
        print(f"✅ [Worker] STT 모델 로드 완료 (Device: {config['stt_device']})")

        # 2. Llama.cpp 로드 (GPU Offload)
        from llama_cpp import Llama
        llm_model = Llama(
            model_path=config['llm_model_path'],
            n_gpu_layers=config['llm_n_gpu_layers'], # -1 (전체 GPU 로드)
            n_ctx=1024,
            verbose=False
        )
        print(f"✅ [Worker] LLM 모델 로드 완료 (GPU Layers: {config['llm_n_gpu_layers']})")

    except Exception as e:
        print(f"❌ [Worker] 모델 로딩 실패: {e}")
        return

    print("🚀 [Worker] 추론 준비 완료! (GPU Running)")

    while True:
        try:
            task = input_queue.get()
            if task is None: break 

            user_id, audio_bytes, source_lang = task
            start_time = time.time()

            # --- 1. STT ---
            audio_segment = AudioSegment(
                data=audio_bytes, sample_width=2, frame_rate=48000, channels=2
            ).set_frame_rate(16000).set_channels(1)
            
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            segments, _ = stt_model.transcribe(
                wav_io, 
                beam_size=5, 
                language=source_lang 
            )
            original_text = " ".join([s.text for s in segments]).strip()

            if not original_text: continue

            # --- 2. LLM ---
            prompt = f"""<|im_start|>system
You are a professional translator. Translate the user input into natural Korean.<|im_end|>
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
            
            total_time = time.time() - start_time
            
            output_queue.put({
                "user_id": user_id,
                "original": original_text,
                "translated": translated_text,
                "time": total_time
            })

        except Exception as e:
            print(f"⚠️ [Worker] 추론 에러: {e}")

# ==========================================
# 🤖 [Process 1] 디스코드 봇
# ==========================================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

task_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()
active_channel = None

class LocalVADSink(discord.sinks.Sink):
    def __init__(self, task_queue, lang_code=None, filters=None):
        if filters is None: filters = discord.sinks.default_filters
        super().__init__(filters=filters)
        self.task_queue = task_queue
        self.lang_code = lang_code
        self.user_data = {}
        self.SILENCE_THRESHOLD = 1000
        self.SILENCE_LIMIT = 0.5 

    def get_user_data(self, user):
        if user not in self.user_data:
            self.user_data[user] = {"buffer": bytearray(), "silence_start": None, "is_speaking": False}
        return self.user_data[user]

    @discord.sinks.Filters.container
    def write(self, data, user):
        ud = self.get_user_data(user)
        try: rms = audioop.rms(data, 2)
        except: rms = 0

        if rms > self.SILENCE_THRESHOLD:
            ud["silence_start"] = None
            ud["is_speaking"] = True
        else:
            if ud["silence_start"] is None:
                ud["silence_start"] = time.time()

        ud["buffer"] += data
        now = time.time()

        if (ud["is_speaking"] and ud["silence_start"] is not None and (now - ud["silence_start"]) > self.SILENCE_LIMIT):
            if len(ud["buffer"]) > 30000: 
                self.task_queue.put((user, bytes(ud["buffer"]), self.lang_code))
            ud["buffer"] = bytearray()
            ud["is_speaking"] = False
            ud["silence_start"] = None

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    bot.loop.create_task(check_results())

async def check_results():
    while True:
        try:
            while not result_queue.empty():
                res = result_queue.get_nowait()
                msg = f"⚡ **{res['translated']}**\n└ `({res['original']})` [⏱️ {res['time']:.2f}s]"
                if active_channel:
                    await active_channel.send(msg)
            await asyncio.sleep(0.01) # GPU는 빠르니까 체크 주기도 짧게
        except Exception as e:
            print(f"Result Loop Error: {e}")
            await asyncio.sleep(1)

@bot.command("translate")
async def translate(ctx, lang: str = "auto"):
    global active_channel
    lang_code = LANG_MAP.get(lang.lower())
    if not lang_code and lang != "auto":
        return await ctx.send(f"❌ 지원하지 않는 언어입니다.")

    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        active_channel = ctx.channel
        ctx.voice_client.start_recording(LocalVADSink(task_queue, lang_code), finished_callback, ctx.channel)
        target = "자동 감지" if lang == "auto" else lang
        await ctx.send(f"🚀 **GPU 가속 통역 시작** (설정: {target})\n엔비디아 쿠다 엔진이 가동되었습니다.")
    else:
        await ctx.send("음성 채널에 먼저 들어가주세요.")

async def finished_callback(sink, channel, *args):
    await channel.send("⏹️ 종료됨.")

@bot.command("leave")
async def leave(ctx):
    if ctx.voice_client:
        ctx.voice_client.stop_recording()
        await ctx.voice_client.disconnect()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    worker = multiprocessing.Process(target=inference_worker, args=(task_queue, result_queue, MODEL_CONFIG))
    worker.daemon = True 
    worker.start()
    if DISCORD_BOT_TOKEN:
        bot.run(DISCORD_BOT_TOKEN)