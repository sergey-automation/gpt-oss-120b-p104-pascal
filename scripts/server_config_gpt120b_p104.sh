# =========================================================
# server_config_gpt120b_p104.sh


MODEL="$HOME/workspace/models/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

CONTEXTS_DIR="$HOME/workspace/projects/llama.cpp/contexts_gpt_tokenizer"
OUTPUT_DIR="$HOME/workspace/projects/llama.cpp/bench_reports_gpt-120b"

TENSOR_SPLIT="2/4/4/4/4/4/4/4/3/3"

UBATCH=1024
BATCH=2048

CTX_SIZE=131072

HOST="127.0.0.1"
PORT=8081

PARALLEL=2
CTK=f16


