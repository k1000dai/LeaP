#!/usr/bin/env bash
# Prepare the LeaP venv on robot_dev6 for the multimodal steering campaign. Idempotent; run ON the remote.
# Steps: (1) install the embedder/TTS packages without touching the simulator stack, (2) fetch the Piper
# voice, (3) pre-download google/embeddinggemma-2, (4) embed text/image/audio once on an idle GPU,
# (5) synthesise one wav with Piper. Every step is skipped when its result already exists.
# Usage: bash scripts/remote_setup_steer.sh
# Env:   LEAP_ROOT (~/codes/LeaP), PROBE_GPU (least-used GPU), SKIP_PROBE (unset), VOICE (en_US-lessac-medium),
#        VOICE_URL (directory of the voice files on huggingface), EMBEDDER (google/embeddinggemma-2)
set -euo pipefail
LEAP_ROOT="${LEAP_ROOT:-$HOME/codes/LeaP}"
VOICE="${VOICE:-en_US-lessac-medium}"
EMBEDDER="${EMBEDDER:-google/embeddinggemma-2}"
PACKAGES=("transformers>=5.18" "sentence-transformers>=6.1" soundfile librosa piper-tts)
PINNED='^(torch|torchvision|numpy|sapien|warp-lang|pytorch3d)$'
log() { echo "[$(date '+%F %T')] $*"; }
trap 'log "SETUP FAILED at line $LINENO"' ERR
cd "$LEAP_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs data/voices
PROBE_DIR="$LEAP_ROOT/logs/steer-setup-probe"; mkdir -p "$PROBE_DIR"

versions() { python - <<'EOF'
import importlib
for name in ("torch", "torchvision", "numpy", "sapien", "warp"):
    try:
        print(name, importlib.import_module(name).__version__)
    except Exception as exc:  # noqa: BLE001
        print(name, "MISSING", exc)
EOF
}

log "=== (1) packages"
BEFORE=$(versions); echo "$BEFORE"
if grep -q "^torchvision MISSING" <<< "$BEFORE"; then
  TORCH=$(python -c "import torch;print(torch.__version__)")
  log "torchvision missing; installing the cu128 build matching torch $TORCH without touching torch"
  uv pip install --no-deps "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
fi
DRY=$(uv pip install --dry-run "${PACKAGES[@]}" 2>&1); echo "$DRY"
# " - name==ver" lines are removals, " + name==ver" additions; either on a pinned package aborts.
TOUCHED=$(grep -E '^ [-+] ' <<< "$DRY" | sed -E 's/^ [-+] ([A-Za-z0-9_.-]+)==.*/\1/' | grep -E "$PINNED" || true)
if [[ -n "$TOUCHED" ]]; then
  log "ABORT: the install would change pinned package(s): $TOUCHED"; exit 1
fi
if echo "$DRY" | grep -q "Would make no changes"; then
  log "packages already installed"
else
  uv pip install "${PACKAGES[@]}"
fi
AFTER=$(versions); echo "$AFTER"
[[ "$BEFORE" == "$AFTER" ]] || { log "ABORT: pinned versions changed"; diff <(echo "$BEFORE") <(echo "$AFTER") || true; exit 1; }
python -c "import transformers, sentence_transformers, soundfile, librosa, piper, huggingface_hub; \
print('transformers', transformers.__version__, 'sentence-transformers', sentence_transformers.__version__, \
'huggingface-hub', huggingface_hub.__version__, 'soundfile', soundfile.__version__, 'librosa', librosa.__version__)"

log "=== (2) Piper voice $VOICE -> data/voices"
BASE="${VOICE_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium}"
for FILE in "$VOICE.onnx" "$VOICE.onnx.json"; do
  if [[ -s "data/voices/$FILE" ]]; then log "data/voices/$FILE present"; continue; fi
  curl -fL --retry 3 -o "data/voices/$FILE.part" "$BASE/$FILE"
  mv "data/voices/$FILE.part" "data/voices/$FILE"
done
python -c "import json,sys;c=json.load(open(sys.argv[1]));print('voice sample_rate', c['audio']['sample_rate'])" "data/voices/$VOICE.onnx.json"
ls -l data/voices

log "=== (3) pre-download $EMBEDDER into the HF cache"
python - "$EMBEDDER" <<'EOF'
import sys, time
from sentence_transformers import SentenceTransformer
t0 = time.time()
model = SentenceTransformer(sys.argv[1], device="cpu")
print(f"loaded on cpu in {time.time() - t0:.1f}s; max_seq_length={model.max_seq_length}; dim={model.get_sentence_embedding_dimension()}")
EOF

if [[ -n "${SKIP_PROBE:-}" ]]; then log "SKIP_PROBE set; done"; log "SETUP OK"; exit 0; fi
log "=== (4) GPU probe"
if [[ -z "${PROBE_GPU:-}" ]]; then
  PROBE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
log "probing on GPU $PROBE_GPU"
CUDA_VISIBLE_DEVICES="$PROBE_GPU" python - "$EMBEDDER" "$PROBE_DIR" <<'EOF'
import json, sys, time
import numpy as np, soundfile as sf, torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

name, out = sys.argv[1], sys.argv[2]
t0 = time.time()
model = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
print(f"loaded on cuda in {time.time() - t0:.1f}s")
texts = ["place the bottle to the left of the cup", "place the bottle to the right of the cup"]
img = Image.new("RGB", (320, 240), "white"); draw = ImageDraw.Draw(img)
draw.ellipse((100, 80, 160, 140), outline="red", width=3); draw.line((130, 110, 250, 110), fill="blue", width=3)
img.save(f"{out}/probe.png")
rate = 16000; t = np.arange(rate) / rate
sf.write(f"{out}/probe.wav", (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
t0 = time.time()
text = model.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True)
image = model.encode([{"text": "<|image|>", "image": [f"{out}/probe.png"]}], normalize_embeddings=True)
audio = model.encode([{"text": "<|audio|>", "audio": [f"{out}/probe.wav"]}], normalize_embeddings=True)
elapsed = time.time() - t0
cos = lambda a, b: float(np.dot(a, b))
report = {"gpu": torch.cuda.get_device_name(0), "text_shape": list(text.shape), "image_shape": list(image.shape),
          "audio_shape": list(audio.shape), "cos_left_right": cos(text[0], text[1]),
          "cos_text_image": cos(text[0], image[0]), "cos_text_audio": cos(text[0], audio[0]),
          "encode_seconds": elapsed, "gpu_mem_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
          "gpu_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
print(json.dumps(report, indent=2))
json.dump(report, open(f"{out}/probe.json", "w"), indent=2)
EOF

log "=== (5) Piper synthesis"
python -m piper --help > /dev/null
echo "Place the bottle to the left of the cup." | python -m piper -m "data/voices/$VOICE.onnx" -f "$PROBE_DIR/piper.wav"
python -c "import soundfile,sys;d,sr=soundfile.read(sys.argv[1]);print('piper wav', d.shape, sr, 'Hz', f'{len(d)/sr:.2f}s')" "$PROBE_DIR/piper.wav"
log "probe files under $PROBE_DIR"
log "SETUP OK"
