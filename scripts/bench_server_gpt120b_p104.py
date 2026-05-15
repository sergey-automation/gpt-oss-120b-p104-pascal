import os, time, csv, signal, subprocess, threading, requests, re
from pathlib import Path
from datetime import datetime

# NOTE:
# GPU load separation between prefill/decode stages is approximate.
# Stage boundaries are estimated from llama-server timing fields:
#   prompt_ms       -> prefill
#   predicted_ms    -> decode
#
# Actual GPU activity may overlap slightly because:
# - HTTP request timing includes non-GPU overhead
# - scheduling/synchronization may continue after prompt_ms
# - nvidia-smi polling is discrete and asynchronous
#
# The goal is comparative benchmarking, not cycle-accurate GPU profiling.



def load_cfg(path="server_config_gpt120b_p104.sh"):
    cfg = {}
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = os.path.expandvars(v.strip().strip('"').strip("'"))

    defaults = {
        # Main
        "MODEL": "$HOME/workspace/models/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
        "CONTEXTS_DIR": "$HOME/workspace/projects/llama.cpp/contexts_gpt_tokenizer",
        "OUTPUT_DIR": "$HOME/workspace/projects/llama.cpp/bench_reports_gpt-120b",

        # Server
        "HOST": "127.0.0.1",
        "PORT": "8081",
        "NGL": "99",
        "CTX_SIZE": "131072",
        "N_GEN": "128",
        "BATCH": "1024",
        "UBATCH": "512",
        "CTK": "f16",
        "SPLIT_MODE": "layer",
        "TENSOR_SPLIT": "2/4/4/4/4/4/4/4/3/3"
,
        "PARALLEL": "1",
        "TEMPERATURE": "0.0",
        "MONITOR_INTERVAL": "0.2",

        # Debug env only, not command args
        "GGML_SCHED_DEBUG": "2",
        "GGML_CUDA_DEBUG": "0",
    }

    for k, v in defaults.items():
        cfg.setdefault(k, os.path.expandvars(v))

    return cfg

cfg = load_cfg()

def nvidia_sample():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,power.draw,memory.used",
        "--format=csv,noheader,nounits"
    ], text=True)
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        rows.append({
            "gpu": int(p[0]),
            "name": p[1],
            "bus": p[2],
            "gen": p[3],
            "width": p[4],
            "util": float(p[5]),
            "power": float(p[6]),
            "mem": float(p[7]),
            "t": time.time(),
        })
    return rows

class Monitor:
    def __init__(self, interval):
        self.interval = float(interval)
        self.samples = []
        self.stop_flag = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=2)

    def run(self):
        while not self.stop_flag:
            try:
                self.samples.extend(nvidia_sample())
            except Exception:
                pass
            time.sleep(self.interval)

def avg_gpu(samples, t0, t1):
    selected = [s for s in samples if t0 <= s["t"] <= t1]
    out = {}
    for gpu in sorted(set(s["gpu"] for s in selected)):
        ss = [s for s in selected if s["gpu"] == gpu]
        if not ss:
            continue
        out[gpu] = {
            "name": ss[-1]["name"],
            "pcie": f"Gen{ss[-1]['gen']} x{ss[-1]['width']}",
            "util_avg": sum(x["util"] for x in ss) / len(ss),
            "util_max": max(x["util"] for x in ss),
            "power_avg": sum(x["power"] for x in ss) / len(ss),
            "power_max": max(x["power"] for x in ss),
            "mem_max": max(x["mem"] for x in ss),
            "samples": len(ss),
        }
    return out

def fmt(x, nd=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return str(x)

def start_server(out_dir):
    subprocess.run("pkill -f llama-server || true", shell=True)

    log_path = out_dir / "llama_server.log"
    log = open(log_path, "w", encoding="utf-8")

    # ВАЖНО:
    # Только проверенные server-флаги из исходного алгоритма.
    # Не добавляем bench-only параметры: -ctv, -t, -C, --poll, -ncmoe, -nkvo, --cache-ram.
    cmd = [
        "./build/bin/llama-server",
        "-m", cfg["MODEL"],
        "--host", cfg["HOST"],
        "--port", str(cfg["PORT"]),
        "-ngl", str(cfg["NGL"]),
        "-c", str(cfg["CTX_SIZE"]),
        "-b", str(cfg["BATCH"]),
        "-ub", str(cfg["UBATCH"]),
        "-np", str(cfg["PARALLEL"]),
        "-ctk", cfg["CTK"],
        "-sm", cfg["SPLIT_MODE"],
        "-ts", cfg["TENSOR_SPLIT"],
    ]

    env = os.environ.copy()
    env["GGML_SCHED_DEBUG"] = cfg.get("GGML_SCHED_DEBUG", "2")

    if cfg.get("GGML_CUDA_DEBUG", "0") != "0":
        env["GGML_CUDA_DEBUG"] = cfg["GGML_CUDA_DEBUG"]

    p = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    url = f"http://{cfg['HOST']}:{cfg['PORT']}/v1/models"
    for _ in range(920):
        if p.poll() is not None:
            log.close()
            try:
                tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
                msg = "\n".join(tail)
            except Exception:
                msg = ""
            raise RuntimeError("llama-server завершился при запуске. Последние строки лога:\n" + msg)

        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return p, log_path, cmd
        except Exception:
            pass
        time.sleep(1)

    p.terminate()
    log.close()
    raise RuntimeError(f"llama-server не запустился за 920 секунд. Лог: {log_path}")

def request_one(prompt):
    url = f"http://{cfg['HOST']}:{cfg['PORT']}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": int(cfg["N_GEN"]),
        "temperature": float(cfg["TEMPERATURE"]),
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "cache_prompt": False,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=3600)
    t1 = time.time()
    r.raise_for_status()
    return t0, t1, r.json()

def prompt_size_from_name(path):
    m = re.search(r"ctx_(?:\d+_)?(\d+)", path.name)
    return int(m.group(1)) if m else 0

def main():
    out_dir = Path(cfg["OUTPUT_DIR"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = out_dir / "report.md"
    raw_csv = out_dir / "raw_results.csv"

    server, server_log, server_cmd = start_server(out_dir)

    ctx_files = sorted(Path(cfg["CONTEXTS_DIR"]).glob("ctx_*.txt"), key=prompt_size_from_name)
    if not ctx_files:
        server.send_signal(signal.SIGINT)
        raise RuntimeError(f"Нет файлов ctx_*.txt в {cfg['CONTEXTS_DIR']}")

    results = []

    try:
        for ctx in ctx_files:
            text = ctx.read_text(encoding="utf-8", errors="ignore")

            mon = Monitor(cfg["MONITOR_INTERVAL"])
            mon.start()

            t0, t1, data = request_one(text)

            mon.stop()

            timings = data.get("timings", {})
            usage = data.get("usage", {})

            prompt_ms = float(timings.get("prompt_ms", 0))
            decode_ms = float(timings.get("predicted_ms", 0))
            prefill_tps = float(timings.get("prompt_per_second", 0))
            decode_tps = float(timings.get("predicted_per_second", 0))

            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            if prompt_tokens is None:
                prompt_tokens = data.get("tokens_evaluated")
            if completion_tokens is None:
                completion_tokens = data.get("tokens_predicted")
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            pre_t0 = t0
            pre_t1 = t0 + prompt_ms / 1000.0
            dec_t0 = pre_t1
            dec_t1 = t1

            pre = avg_gpu(mon.samples, pre_t0, pre_t1)
            dec = avg_gpu(mon.samples, dec_t0, dec_t1)

            r = {
                "file": ctx.name,
                "prompt_target": prompt_size_from_name(ctx),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prefill_ms": prompt_ms,
                "prefill_tps": prefill_tps,
                "decode_ms": decode_ms,
                "decode_tps": decode_tps,
                "wall_ms": (t1 - t0) * 1000.0,
                "pre": pre,
                "dec": dec,
            }
            results.append(r)

            print(f"{ctx.name}: prefill={prefill_tps:.2f} tok/s decode={decode_tps:.2f} tok/s")

    finally:
        try:
            server.send_signal(signal.SIGINT)
        except Exception:
            pass

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["file", "prompt_tokens", "completion_tokens", "prefill_ms", "prefill_tok_s", "decode_ms", "decode_tok_s", "wall_ms"])
        for r in results:
            w.writerow([r["file"], r["prompt_tokens"], r["completion_tokens"], r["prefill_ms"], r["prefill_tps"], r["decode_ms"], r["decode_tps"], r["wall_ms"]])

    with open(report, "w", encoding="utf-8") as f:
        f.write("# llama-server context benchmark report\n\n")

        f.write("## Test header\n\n")
        for k in ["MODEL", "NGL", "CTX_SIZE", "N_GEN", "BATCH", "UBATCH", "CTK", "SPLIT_MODE", "TENSOR_SPLIT", "PARALLEL", "TEMPERATURE"]:
            f.write(f"- {k}: `{cfg[k]}`\n")
        f.write(f"- llama_server_log: `{server_log}`\n\n")

        f.write("## Server command\n\n```bash\n")
        f.write(" ".join(server_cmd))
        f.write("\n```\n\n")

        f.write("## Summary\n\n")
        f.write("| File | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Wall s |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['file']} "
                f"| {r['prompt_tokens']} "
                f"| {r['completion_tokens']} "
                f"| {fmt(r['prefill_tps'])} "
                f"| {fmt(r['prefill_ms']/1000.0)} "
                f"| {fmt(r['decode_tps'])} "
                f"| {fmt(r['wall_ms']/1000.0)} |\n"
            )

        f.write("\n## GPU load by stage\n\n")
        for r in results:
            f.write(f"### {r['file']}\n\n")
            f.write("Prefill stage:\n\n")
            f.write("| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |\n")
            f.write("|---:|---|---|---:|---:|---:|---:|---:|\n")
            for gpu, s in r["pre"].items():
                f.write(f"| {gpu} | {s['name']} | {s['pcie']} | {fmt(s['util_avg'],1)} | {fmt(s['util_max'],1)} | {fmt(s['power_avg'],1)} | {fmt(s['power_max'],1)} | {fmt(s['mem_max'],0)} |\n")

            f.write("\nDecode stage:\n\n")
            f.write("| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |\n")
            f.write("|---:|---|---|---:|---:|---:|---:|---:|\n")
            for gpu, s in r["dec"].items():
                f.write(f"| {gpu} | {s['name']} | {s['pcie']} | {fmt(s['util_avg'],1)} | {fmt(s['util_max'],1)} | {fmt(s['power_avg'],1)} | {fmt(s['power_max'],1)} | {fmt(s['mem_max'],0)} |\n")
            f.write("\n")

    print("REPORT:", report)
    print("CSV:", raw_csv)

if __name__ == "__main__":
    main()
