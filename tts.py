import asyncio
import os

# ==========================================
# [설정] 가상 케이블 장치 번호
# ==========================================
VIRTUAL_CABLE_ID = 5 
# ==========================================

# 텍스트와 성우 리스트
playlist = [
    # 1. 영어
    (
        "Hello, today I present the \"Ultra-Low Latency Local AI Translation System\" powered by GPU acceleration. "
        "This system operates solely on local resources without relying on external APIs. "
        "This approach eliminates API costs, strengthens data security, and minimizes network latency.",
        "en-US-AriaNeural"
    ),
    # 2. 일본어
    (
        "性能最適化のために2つの核心技術を適用しました。"
        "第一に、マルチプロセッシング・アーキテクチャを導入し、AI演算中もボットが停止しないよう設計しました。"
        "第二に、Faster-Whisperエンジンと4ビット量子化技術により、メモリ使用量を削減しつつ推論速度を4倍以上に向上させました。",
        "ja-JP-NanamiNeural"
    ),
    # 3. 중국어
    (
        "正如演示所示，用户的语音输入由本地GPU即时处理，提供无延迟的实时翻译。"
        "本项目证明了即使不依赖商业云服务，也能在个人电脑环境中成功构建高性能的AI基础设施。",
        "zh-CN-XiaoxiaoNeural"
    )
]

async def prepare_and_play_manual():
    import edge_tts
    import sounddevice as sd
    import soundfile as sf
    print("=== [준비 단계] 오디오 미리 생성 중... ===")
    
    # 생성된 오디오 데이터를 담아둘 리스트
    audio_queue = []

    # 1. 모든 오디오 미리 생성 (Pre-loading)
    for i, (text, voice) in enumerate(playlist):
        print(f"   ⏳ {i+1}번 파트 생성 중 ({voice})...")
        filename = f"temp_step_{i}.wav"
        
        # 파일 생성
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)
        
        # 데이터를 메모리로 로드
        data, fs = sf.read(filename, dtype='float32')
        
        # (데이터, 샘플레이트, 텍스트내용) 튜플로 저장
        audio_queue.append((data, fs, text))
        
        # 임시 파일 즉시 삭제 (메모리에 올렸으므로 필요 없음)
        if os.path.exists(filename):
            os.remove(filename)
            
    print("\n=== [준비 완료] 엔터를 누르면 순서대로 재생합니다 ===")
    print("-------------------------------------------------------")

    # 2. 사용자 입력에 따라 재생 (Step-by-Step)
    for index, (data, fs, text_content) in enumerate(audio_queue):
        # 짧게 요약된 텍스트 보여주기
        preview_text = text_content[:50] + "..." if len(text_content) > 50 else text_content
        
        # 사용자 입력 대기
        input(f"[{index+1}/{len(audio_queue)}] 엔터를 누르면 재생합니다: \"{preview_text}\"")
        
        print(f"   ▶ 재생 중... (장치 ID: {VIRTUAL_CABLE_ID})")
        
        # 재생
        sd.play(data, fs, device=VIRTUAL_CABLE_ID)
        sd.wait() # 재생이 끝날 때까지 대기 (중복 재생 방지)
        print("   ✅ 재생 완료\n")

    print("=== 모든 프레젠테이션이 종료되었습니다 ===")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        
    loop.run_until_complete(prepare_and_play_manual())
