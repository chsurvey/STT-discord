import gc
import json
import os
import statistics
import threading
import time
import inspect

# NOTE: The package 'nvidia-ml-py' is imported as 'pynvml'.
import pynvml
import psutil
import torch
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_PATH = os.path.join(ROOT_DIR, "ablation_results.json")

# Update these paths if your filenames differ
TRANSFORMERS_BASE = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct")
LLAMA_Q8_0_PATH = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q8_0.gguf")
LLAMA_Q4_K_M_PATH = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

TEXT_SNIPPET = (
    "Hello, today I present the \"Ultra-Low Latency Local AI Translation System\" powered by GPU acceleration. "
    "This system operates solely on local resources without relying on external APIs. "
    "This approach eliminates API costs, strengthens data security, and minimizes network latency."
)

PROMPT_TEMPLATE = """<|im_start|>system
You are a professional translator. Translate the user input into natural Korean.<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""

STOP_STRINGS = ["<|im_end|>", "\n"]
MAX_TOKENS = 128
TEMPERATURE = 0.0
N_SAMPLES = 10
WARMUP_RUNS = 1
N_CTX = 4096
N_GPU_LAYERS = 9999
MB = 1024 * 1024


def build_prompt(text):
    return PROMPT_TEMPLATE.format(input_text=text)


def truncate_at_stop(text):
    cut = len(text)
    for s in STOP_STRINGS:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def get_process_vram_mb(handle, pid):
    fn_v2 = getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None)
    if fn_v2 is not None:
        procs = fn_v2(handle)
    else:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    if not procs:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            used = p.usedGpuMemory
            if used is None:
                return 0.0
            return used / MB
    return 0.0


def run_with_monitor(call_fn, handle, pid):
    stop = threading.Event()
    peaks = {"cpu_percent": 0.0, "vram_mb": 0.0, "rss_mb": 0.0}

    def monitor():
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)
        while not stop.is_set():
            cpu = proc.cpu_percent(interval=0.05)
            rss = proc.memory_info().rss / MB
            vram = get_process_vram_mb(handle, pid)
            if cpu > peaks["cpu_percent"]:
                peaks["cpu_percent"] = cpu
            if rss > peaks["rss_mb"]:
                peaks["rss_mb"] = rss
            if vram > peaks["vram_mb"]:
                peaks["vram_mb"] = vram

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    result = call_fn()
    latency = time.perf_counter() - start
    stop.set()
    thread.join()
    return result, latency, peaks["cpu_percent"], peaks["vram_mb"], peaks["rss_mb"]


def run_transformers_case(prompt, handle, pid):
    print("Case: Transformers base (baseline)")
    print("Loading Transformers model...")
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMERS_BASE, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        TRANSFORMERS_BASE,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    torch.cuda.synchronize()
    torch_load_allocated_mb = torch.cuda.memory_allocated() / MB
    torch_load_reserved_mb = torch.cuda.memory_reserved() / MB
    print(
        "CUDA mem allocated/reserved: "
        f"{torch_load_allocated_mb:.1f} MB / "
        f"{torch_load_reserved_mb:.1f} MB"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_rss_mb = psutil.Process(os.getpid()).memory_info().rss / MB
    base_vram_nvml_mb = get_process_vram_mb(handle, pid)
    base_vram_torch_reserved_mb = torch_load_reserved_mb
    base_vram_mb = max(base_vram_nvml_mb, base_vram_torch_reserved_mb)
    print(f"NVML VRAM after load: {base_vram_nvml_mb:.1f} MB")

    def generate_once():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        peak_alloc_mb = torch.cuda.max_memory_allocated() / MB
        peak_reserved_mb = torch.cuda.max_memory_reserved() / MB
        return gen_ids, inputs["input_ids"].shape[1], peak_alloc_mb, peak_reserved_mb

    print("Warming up...")
    for _ in range(WARMUP_RUNS):
        _ = generate_once()
    torch.cuda.synchronize()
    base_vram_torch_reserved_mb = max(
        base_vram_torch_reserved_mb, torch.cuda.memory_reserved() / MB
    )
    base_rss_mb = max(
        base_rss_mb, psutil.Process(os.getpid()).memory_info().rss / MB
    )
    base_vram_nvml_mb = max(base_vram_nvml_mb, get_process_vram_mb(handle, pid))
    base_vram_mb = max(base_vram_mb, base_vram_nvml_mb, base_vram_torch_reserved_mb)

    print("Running samples...")
    samples = []
    for i in range(N_SAMPLES):
        (
            (gen_ids, prompt_len, peak_alloc_mb, peak_reserved_mb),
            latency,
            cpu_peak,
            vram_peak,
            rss_peak,
        ) = run_with_monitor(generate_once, handle, pid)
        gen_tokens = int(gen_ids.shape[-1] - prompt_len)
        tokens_per_sec = gen_tokens / latency
        samples.append(
            {
                "sample_idx": i,
                "latency_s": latency,
                "gen_tokens": gen_tokens,
                "tokens_per_sec": tokens_per_sec,
                "cpu_peak_percent": cpu_peak,
                "vram_peak_mb": vram_peak,
                "rss_peak_mb": rss_peak,
                "torch_peak_allocated_mb": peak_alloc_mb,
                "torch_peak_reserved_mb": peak_reserved_mb,
            }
        )

    latencies = [s["latency_s"] for s in samples]
    tokens_per_sec_vals = [s["tokens_per_sec"] for s in samples]
    cpu_peaks = [s["cpu_peak_percent"] for s in samples]
    vram_peaks = [s["vram_peak_mb"] for s in samples]
    rss_peaks = [s["rss_peak_mb"] for s in samples]
    torch_peak_allocated_mb_max = max(s["torch_peak_allocated_mb"] for s in samples)
    torch_peak_reserved_mb_max = max(s["torch_peak_reserved_mb"] for s in samples)
    result = {
        "name": "transformers_base",
        "model_path": TRANSFORMERS_BASE,
        "base_rss_mb": base_rss_mb,
        "base_vram_mb": base_vram_mb,
        "base_vram_nvml_mb": base_vram_nvml_mb,
        "base_vram_torch_reserved_mb": base_vram_torch_reserved_mb,
        "rss_peak_mb_max": max(rss_peaks),
        "vram_peak_mb": max(
            [base_vram_mb] + vram_peaks + [torch_peak_reserved_mb_max]
        ),
        "torch_peak_allocated_mb_max": torch_peak_allocated_mb_max,
        "torch_peak_reserved_mb_max": torch_peak_reserved_mb_max,
        "cpu_peak_percent_max": max(cpu_peaks),
        "latency_p50_s": statistics.median(latencies),
        "tokens_per_sec_p50": statistics.median(tokens_per_sec_vals),
        "samples": samples,
    }

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return result


def run_llama_case(case_name, model_path, prompt, handle, pid):
    print(f"Case: {case_name}")
    print("Loading llama.cpp model...")
    
    # Attempt to offload all layers
    # Force use_mmap=False to make VRAM usage visible immediately
    llama_kwargs = {
        "model_path": model_path,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "main_gpu": 0,
        "verbose": True,  # ENABLED: Look for 'offloading' logs in console!
        "use_mmap": False, 
        "n_batch": 512,
    }

    # Filter arguments for compatibility
    sig = inspect.signature(Llama.__init__)
    supported = set(sig.parameters.keys())
    final_kwargs = {k: v for k, v in llama_kwargs.items() if k in supported}

    llm = Llama(**final_kwargs)

    base_rss_mb = psutil.Process(os.getpid()).memory_info().rss / MB
    base_vram_mb = get_process_vram_mb(handle, pid)
    print(f"NVML VRAM after load: {base_vram_mb:.1f} MB")
    
    # Simple check for the user
    if base_vram_mb < 100:
        print("GPU offload worked? NO (VRAM near zero)")
        print("Reinstall llama-cpp-python with CMAKE_ARGS='-DGGML_CUDA=on'")
    else:
        print("GPU offload worked? YES")

    def generate_once():
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            stop=STOP_STRINGS,
        )
        return output["choices"][0]["text"]

    print("Warming up...")
    for _ in range(WARMUP_RUNS):
        _ = generate_once()
    base_rss_mb = max(
        base_rss_mb, psutil.Process(os.getpid()).memory_info().rss / MB
    )
    base_vram_mb = max(base_vram_mb, get_process_vram_mb(handle, pid))

    print("Running samples...")
    samples = []
    for i in range(N_SAMPLES):
        text, latency, cpu_peak, vram_peak, rss_peak = run_with_monitor(
            generate_once, handle, pid
        )
        text = truncate_at_stop(text).strip()
        token_bytes = text.encode("utf-8") if text else b""
        gen_tokens = len(llm.tokenize(token_bytes, add_bos=False))
        tokens_per_sec = gen_tokens / latency
        samples.append(
            {
                "sample_idx": i,
                "latency_s": latency,
                "gen_tokens": gen_tokens,
                "tokens_per_sec": tokens_per_sec,
                "cpu_peak_percent": cpu_peak,
                "vram_peak_mb": vram_peak,
                "rss_peak_mb": rss_peak,
            }
        )

    latencies = [s["latency_s"] for s in samples]
    tokens_per_sec_vals = [s["tokens_per_sec"] for s in samples]
    cpu_peaks = [s["cpu_peak_percent"] for s in samples]
    vram_peaks = [s["vram_peak_mb"] for s in samples]
    rss_peaks = [s["rss_peak_mb"] for s in samples]
    result = {
        "name": case_name,
        "model_path": model_path,
        "base_rss_mb": base_rss_mb,
        "base_vram_mb": base_vram_mb,
        "rss_peak_mb_max": max(rss_peaks),
        "vram_peak_mb": max([base_vram_mb] + vram_peaks),
        "cpu_peak_percent_max": max(cpu_peaks),
        "latency_p50_s": statistics.median(latencies),
        "tokens_per_sec_p50": statistics.median(tokens_per_sec_vals),
        "samples": samples,
    }

    del llm
    gc.collect()
    return result


def main():
    print("Starting ablation (LLM-only latency + resource usage)...")
    prompt = build_prompt(TEXT_SNIPPET)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pid = os.getpid()

    results = {
        "config": {
            "sample_text": TEXT_SNIPPET,
            "prompt_template": PROMPT_TEMPLATE,
            "stop_strings": STOP_STRINGS,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "n_samples": N_SAMPLES,
            "warmup_runs": WARMUP_RUNS,
            "n_ctx": N_CTX,
            "n_gpu_layers": N_GPU_LAYERS,
        },
        "cases": [],
    }

    results["cases"].append(run_llama_case("llama_cpp_q8_0", LLAMA_Q8_0_PATH, prompt, handle, pid))
    results["cases"].append(run_llama_case("llama_cpp_q4_k_m", LLAMA_Q4_K_M_PATH, prompt, handle, pid))
    results["cases"].append(run_transformers_case(prompt, handle, pid))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pynvml.nvmlShutdown()
    print(f"Done. Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
