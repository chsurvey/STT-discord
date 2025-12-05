import discord
from discord.ext import commands
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import audioop
import time
import logging

# 로그 숨김
logging.getLogger("discord.opus").setLevel(logging.CRITICAL)
logging.getLogger("discord.voice_client").setLevel(logging.CRITICAL)

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

r = sr.Recognizer()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 언어 코드 매핑 (사용자 입력 -> Google/Whisper 호환용)
LANG_MAP = {
    "en": {"google": "en-US", "whisper": "en", "name": "영어"},
    "ja": {"google": "ja-JP", "whisper": "ja", "name": "일본어"},
    "zh": {"google": "zh-CN", "whisper": "zh", "name": "중국어"},
    "es": {"google": "es-ES", "whisper": "es", "name": "스페인어"},
    "fr": {"google": "fr-FR", "whisper": "fr", "name": "프랑스어"},
    "ko": {"google": "ko-KR", "whisper": "ko", "name": "한국어"}
}

class SmartTranslateSink(discord.sinks.Sink):
    def __init__(self, bot, lang_code, filters=None):
        if filters is None:
            filters = discord.sinks.default_filters
        super().__init__(filters=filters)
        
        self.bot = bot
        self.user_data = {}
        
        # 선택된 언어 설정
        self.lang_config = LANG_MAP.get(lang_code, LANG_MAP["en"]) # 기본값 영어
        self.source_lang_name = self.lang_config["name"]
        
        # VAD 설정 (엄격 모드 적용)
        self.SILENCE_THRESHOLD = 1000
        self.SILENCE_LIMIT = 0.5
        self.GOOGLE_INTERVAL = 2.0

    def get_user_data(self, user):
        if user not in self.user_data:
            self.user_data[user] = {
                "buffer": bytearray(),
                "silence_start": None,
                "last_google_time": time.time(),
                "temp_message": None,
                "is_speaking": False,
                "has_spoken": False
            }
        return self.user_data[user]

    @discord.sinks.Filters.container
    def write(self, data, user):
        ud = self.get_user_data(user)
        try:
            rms = audioop.rms(data, 2)
        except: rms = 0

        # VAD 로직
        if rms > self.SILENCE_THRESHOLD:
            ud["has_spoken"] = True
            ud["silence_start"] = None
            if not ud["is_speaking"]:
                # print(f"🗣️ [{self.source_lang_name}] User:{user} Speaking...")
                ud["is_speaking"] = True
        else:
            if ud["silence_start"] is None:
                ud["silence_start"] = time.time()
            if ud["is_speaking"]:
                # print(f"🤫 [{self.source_lang_name}] Silence detected.")
                ud["is_speaking"] = False

        ud["buffer"] += data
        now = time.time()

        # Case A: Whisper + GPT 번역 (문장 종료)
        if (ud["silence_start"] is not None and 
            (now - ud["silence_start"]) > self.SILENCE_LIMIT):
            
            # 짧은 대답도 놓치지 않게 0.2초 분량
            if len(ud["buffer"]) > 38000: 
                if ud["has_spoken"]:
                    audio_to_process = bytes(ud["buffer"])
                    asyncio.run_coroutine_threadsafe(
                        self.process_translate_full(user, audio_to_process),
                        self.bot.loop
                    )
            
            ud["buffer"] = bytearray()
            ud["silence_start"] = None
            ud["has_spoken"] = False

        # Case B: Google STT (중간 확인 - 원문 표시)
        elif (ud["has_spoken"] and 
              ud["silence_start"] is None and 
              (now - ud["last_google_time"]) > self.GOOGLE_INTERVAL and
              len(ud["buffer"]) > 100000):
            
            ud["last_google_time"] = now
            audio_snapshot = bytes(ud["buffer"])
            asyncio.run_coroutine_threadsafe(
                self.process_google_fast(user, audio_snapshot),
                self.bot.loop
            )

    async def process_google_fast(self, user, audio_bytes):
        """Google STT: 해당 언어로 인식하여 원문을 보여줌"""
        try:
            audio_segment = AudioSegment(
                data=audio_bytes, sample_width=2, frame_rate=48000, channels=2
            ).set_frame_rate(16000).set_channels(1)
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                try:
                    # 설정된 외국어로 인식
                    text = r.recognize_google(audio_data, language=self.lang_config["google"])
                    if text.strip():
                        ud = self.user_data[user]
                        # 원문을 보여줌 (번역 전 단계)
                        new_content = f"Listening({self.lang_config['whisper']})... <@{user}>: {text}"
                        
                        if ud["temp_message"]:
                            try: await ud["temp_message"].edit(content=new_content)
                            except: ud["temp_message"] = await self.channel.send(new_content)
                        else:
                            ud["temp_message"] = await self.channel.send(new_content)
                except: pass
        except Exception: pass

    async def process_translate_full(self, user, audio_bytes):
        """Whisper(받아쓰기) -> GPT(한글 번역)"""
        try:
            # 1. 오디오 준비
            audio_segment = AudioSegment(
                data=audio_bytes, sample_width=2, frame_rate=48000, channels=2
            )
            mp3_io = io.BytesIO()
            mp3_io.name = "audio.mp3"
            audio_segment.export(mp3_io, format="mp3")
            mp3_io.seek(0)

            # 2. Whisper로 해당 언어 받아쓰기 (Transcribe)
            def call_whisper():
                return openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=mp3_io,
                    language=self.lang_config["whisper"], # 예: 'en'
                    prompt="No subtitles, no captions. Just the spoken text."
                )
            
            transcript = await asyncio.to_thread(call_whisper)
            original_text = transcript.text.strip()
            
            # 환각 필터링
            triggers = ["자막 제공", "MBC", "시청해", "Subtitles", "Caption"]
            for t in triggers:
                if t in original_text:
                    original_text = original_text.split(t)[0].strip()

            if original_text:
                # 3. GPT-4o-mini (또는 3.5)로 한국어 번역 수행
                def call_gpt_translate():
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini", # 가성비 모델 사용
                        messages=[
                            {"role": "system", "content": "You are a professional translator. Translate the user's input into natural Korean."},
                            {"role": "user", "content": original_text}
                        ]
                    )
                    return response.choices[0].message.content

                translated_text = await asyncio.to_thread(call_gpt_translate)

                # 4. 결과 전송
                ud = self.user_data[user]
                if ud["temp_message"]:
                    try: await ud["temp_message"].delete()
                    except: pass
                    ud["temp_message"] = None
                
                # 포맷: [한글 번역] (원문)
                await self.channel.send(f"✅ <@{user}>: **{translated_text}** \n└ `({original_text})`")

        except Exception as e:
            print(f"Translation Error: {e}")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')

@bot.command("join")
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"✅ **{channel.name}** 채널에 접속했습니다.")
    else:
        await ctx.send("먼저 음성 채널에 접속해주세요.")

@bot.command("record")
async def record(ctx):
    vc = ctx.voice_client
    if not vc:
        return await ctx.send("봇이 음성 채널에 없습니다.")

    # Smart VAD Sink 사용
    sink = SmartTranslateSink(bot)
    sink.channel = ctx.channel 
    
    vc.start_recording(
        sink,
        finished_callback,
        ctx.channel
    )
    
    await ctx.send("🎙️ **스마트 STT 시작!** (말이 끝나면 자동으로 변환합니다)")

async def finished_callback(sink, channel, *args):
    await channel.send("⏹️ 세션 종료.")
    
@bot.command("translate")
async def translate(ctx, lang: str = "en"):
    """
    사용법: !translate [언어코드]
    예시: !translate en (영어), !translate ja (일본어)
    """
    vc = ctx.voice_client
    if not vc:
        return await ctx.send("봇이 음성 채널에 없습니다.")

    # 언어 코드 확인
    lang = lang.lower()
    if lang not in LANG_MAP:
        supported = ", ".join(LANG_MAP.keys())
        return await ctx.send(f"지원하지 않는 언어입니다. 지원 언어: {supported}")

    selected = LANG_MAP[lang]
    sink = SmartTranslateSink(bot, lang)
    sink.channel = ctx.channel 
    
    vc.start_recording(
        sink,
        finished_callback_dummy,
        ctx.channel
    )
    
    await ctx.send(f"🌐 **실시간 통역 시작!** ({selected['name']} -> 한국어)\n말씀하시면 한국어로 번역해 드립니다.")

async def finished_callback_dummy(sink, channel, *args):
    await channel.send("⏹️ 통역 세션 종료.")

@bot.command("stop")
async def stop(ctx):
    vc = ctx.voice_client
    if vc and vc.recording:
        vc.stop_recording()
    await ctx.send("⏹️ 중지됨.")

@bot.command("leave")
async def leave(ctx):
    vc = ctx.voice_client
    if vc:
        if vc.recording:
            vc.stop_recording()
        await vc.disconnect()
        await ctx.send("👋")

bot.run(DISCORD_BOT_TOKEN)