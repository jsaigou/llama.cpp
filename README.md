# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## This fork (jsaigou/llama.cpp, branch `kintsugi`)

This is a working fork of upstream `ggml-org/llama.cpp`, maintained for **Tohil**, a
self-hosted inference host running an AMD Strix Halo (Ryzen AI Max+ 395, `gfx1151`,
128 GB unified memory). The `kintsugi` branch is rebased onto current upstream `master`
periodically (last rebase: 2026-09-01, onto `458681e1d`) and carries two custom patch
sets on top of stock upstream:

### 1. Kintsugi — cross-turn KV cache reuse for hybrid/recurrent architectures

**Status: production, stable.** One commit
(`Kintsugi: cross-turn KV cache reuse for hybrid/recurrent architectures`) that lets
hybrid/recurrent-architecture models (Qwen3.5 MoE, Qwen3.6, qwen4exp, etc.) reuse KV
cache across conversation turns instead of reprocessing the full context every turn.
In production use across multiple models on Tohil.

### 2. Qwen3.8-Flash-Next (`qwen4exp`) MTP speculative decoding

**Status: experimental, hardware-verified but not yet in daily production use.**
Upstream's `qwen4exp` architecture (Qwen3.8-Flash-Next) shipped without its 4B MTP
(multi-token-prediction) draft head — the real performance lever for this model.
This fork assembles the pieces needed to use it, from several in-flight upstream
community PRs (none merged to `ggml-org/llama.cpp` as of this writing):

- [ggml-org/llama.cpp#27836](https://github.com/ggml-org/llama.cpp/pull/27836) —
  `qwen4exp : add NextN/MTP draft head (--spec-type draft-mtp)` (draft, by
  @rmonsurate)
- [crusaderky's detached-head loader fix](https://github.com/crusaderky/llama.cpp/commit/a82a58a57fc307e5cec0dc68db64d143339be4f2) —
  fixes a `blk.0` vs. `blk.38` tensor-naming mismatch that otherwise fails standalone
  draft-model loading
- The `mtp_only` export whitelist fix + hipCUB TOP_K integration
  ([#26592](https://github.com/ggml-org/llama.cpp/pull/26592)) from
  [@drluoto](https://github.com/drluoto)'s
  [`strix-halo-flash-next`](https://github.com/drluoto/llama.cpp/tree/strix-halo-flash-next)
  branch, which this fork's MTP work is based on
- Follow-up correctness/decode fixes from
  [#27941](https://github.com/ggml-org/llama.cpp/pull/27941),
  [#27977](https://github.com/ggml-org/llama.cpp/pull/27977), and
  [#28068](https://github.com/ggml-org/llama.cpp/pull/28068)

**Known requirement — currently unexplained:** on Tohil's ROCm-TheRock 10.1 toolchain,
the hipCUB TOP_K path produces garbage decoded token IDs (`Invalid input batch` /
corrupted output) under HIP graph capture unless launched with
`GGML_CUDA_DISABLE_GRAPHS=1`.

**This is NOT the rocPRIM graph-capture crash discussed in PR #26592's review thread**
(`DeviceSegmentedRadixSort` crashing with `operation not permitted when stream is
capturing` on rocPRIM < 4.4.0 / ROCm < 7.13, per @Geramy's diagnosis there) — that bug
is a hard crash, not silent corruption, and Tohil's toolchain already ships past the
fix: rocPRIM and hipCUB are both **4.5.0** here, and the `is_graph_capture` guard
(`rocprim/device/device_segmented_radix_sort.hpp`) is present in Tohil's headers,
verified directly (not inferred from the version number). Checked PR #27836's full
comment thread and searched `ggml-org/llama.cpp` issues for this exact failure mode
(garbage/invalid decoded token IDs under MTP + HIP graphs) on 2026-09-01 — nobody has
reported it. This looks like a separate, still-undocumented bug, most likely in the
MTP draft/verify graph-capture interaction itself (very fresh, unreviewed draft-PR
code) rather than the rocPRIM library issue. Not yet reported upstream. Standard ROCm
7.1/7.2.3 builds per community reports in the PR thread do not need
`GGML_CUDA_DISABLE_GRAPHS=1`, for what is evidently a different reason than Tohil does.

**Measured on Tohil (2026-09-01), gfx1151/ROCm, `Qwen3.8-Flash-Next-UD-Q3_K_XL` +
[`drluoto/Qwen3.8-Flash-Next-MTP-GGUF`](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF)
Q8_0 draft sidecar, 32K context, greedy decode, 400-token coding completion:**

| Config | Decode speed | vs. baseline |
|---|---|---|
| No speculation | 22.06 t/s | — |
| `--spec-type draft-mtp --spec-draft-n-max 3` | 34.5–34.7 t/s | **+56–57%** |

Draft acceptance ~86% (269–273/307–312 tokens accepted). Output verified coherent and
correct on real coding prompts.

**Not yet done:** multi-turn Kintsugi-cache stress test under MTP, long-context (>32K)
verification, and a production catalog promotion decision. Upstream PR #27836 is still
draft/unmerged — this fork tracks it, not a stable release.

### Updating this fork

Both patch sets are single self-contained commits/short commit chains, kept as a clean
cherry-pick chain on top of upstream `master` (not a long-lived merge) so they can be
replayed onto a newer `master` at any time:

```sh
git checkout -b kintsugi-next upstream/master
git cherry-pick <kintsugi-commit>
git cherry-pick <qwen4exp-mtp-commit-range>
# resolve any conflicts (expect occasional trivial ones as upstream's own
# qwen4exp support evolves — diff carefully, they've so far all been comment/
# already-superseded-test-case conflicts, not semantic ones)
```

### 3. Experimental branch: `experiment/qwen4exp-mtp-ondevice` (2026-09-02)

**Status: findings recorded, not merged to `kintsugi`.** Branch + worktree at
`/opt/tohil/llama.cpp-kintsugi-rocm10`, built as `build-rocm-exp` against the same ROCm-TheRock
10.1 toolchain as `build-rocm-new`. Two commits on top of `kintsugi` @ `189ed9bef`:

1. `server: keep speculative checkpoints on device (ON_DEVICE)` — upstream's
   [ggml-org/llama.cpp#27836](https://github.com/ggml-org/llama.cpp/pull/27836) thread
   (JayToltTech, 2026-08-31) root-caused a 4x MTP slowdown on Vulkan/gfx1151 to the server
   taking a full host-side recurrent-state checkpoint every speculative round (`qwen4exp` is
   classified `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`), and published a 6-line fix
   (`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` at the six `spec_ckpt` call sites in
   `tools/server/server-context.cpp`) that measured 4.33→16.08 t/s at 70k ctx on Vulkan and
   +61% on CUDA. Applied here verbatim from
   [JayToltTech/llama.cpp#1](https://github.com/JayToltTech/llama.cpp/pull/1).
2. `qwen4exp: direct reads for the lazy PLE table` — clean cherry-pick of
   [ggml-org/llama.cpp#28136](https://github.com/ggml-org/llama.cpp/pull/28136) (coder543,
   open). Opt-in via `-lzm on-direct`; inactive in all testing below (not passed).

**Finding 1 — ON_DEVICE has no measurable effect on this hardware.** A/B'd against the
unmodified `build-rocm-new` binary (same commit, same everything else), at both 32K context
(28.8–29.0 t/s both builds, identical 220/345 draft acceptance) and a real ~56K-token context
(18.4–20.0 t/s patched vs. 18.7–19.9 t/s control, identical 177/264 and 179/261 acceptance).
Tohil is a **unified-memory APU** (`GGML_HIP_UMA=ON`) — no PCIe hop between host and device —
architecturally closer to the Metal case in the upstream thread (where the same fix measured
neutral: "the host-path state save is close to free") than to the discrete Vulkan/CUDA cards
where it was a 4–5x win. The fix is correct and harmless to carry, but on this class of hardware
it isn't the lever. 128K was not separately retested given how tightly 32K and 56K already
agreed.

**Finding 2 — the documented `GGML_CUDA_DISABLE_GRAPHS=1` requirement did not reproduce.**
Re-tested the exact condition described in §2 above (garbage token IDs under HIP graph capture
without the flag) on both `build-rocm-new` (unmodified) and `build-rocm-exp`, with the
checkpoint mechanism confirmed actually firing (17 create/restore cycles logged at `-lv 5` in a
150-token generation; thousands more across all runs below) — every run stayed coherent:
- 32K ctx, 2 runs × 2 builds, 400 tokens each: coherent throughout, ~19–22 t/s (vs. ~28.8–29 t/s
  with the flag set — **disabling graphs costs ~30–35% throughput**, so this isn't a free
  workaround).
- 32K ctx, 1500-token generation (887/1279 draft accepted): coherent throughout, no divergence.
- 4-turn conversation exercising Kintsugi cross-turn cache reuse together with MTP: correct
  cross-turn recall, no corruption.

This directly contradicts the "hardware-verified... nobody has reported it" framing of the
requirement above. Possible explanations, none confirmed: an intervening fix already in this
fork's own #27941/#27977/#28068 follow-ups since whatever build first hit the bug; a
driver/thermal state difference between sessions; or a real but rarer trigger condition this
retest didn't happen to hit. **Do not remove `GGML_CUDA_DISABLE_GRAPHS=1` from any production
config on the strength of this alone** — this is one retest, not a disproof — but the throughput
cost is real enough that it's worth a longer soak test before trusting either claim as final.

**Net assessment (2026-09-02):** the model is in a better state than either shelve decision or
the 2026-09-01 entry above suggests — MTP works, is coherent, and the ON_DEVICE fix (harmless,
just not a lever here) plus this retest narrow what's actually still open to exactly one
question: is `GGML_CUDA_DISABLE_GRAPHS=1` still needed on this hardware, or was it already
fixed and the requirement is now stale documentation. Everything else (long-context coherence,
multi-turn + MTP interaction, ROCm 10.1 viability) checked out clean in this pass.

### Finding 2 continued (2026-09-03) — still did not reproduce, across a wider matrix; a
### different, real bug found instead

Pushed further on the same open question, on `build-rocm-eogfix` (kintsugi @ `189ed9bef` plus
the EOG-checkpoint fix above — inert for this purpose, doesn't touch graph capture) with graphs
left enabled throughout (no `GGML_CUDA_DISABLE_GRAPHS`):

- **Single-stream (`-np 1`, matching production), wider draft window, longer generation:**
  `--spec-draft-n-max 8` (vs. production's 3–4), a dense adversarial code-generation prompt
  (full red-black tree implementation, no repetition to hide behind), 3800 tokens in one
  continuous generation, 4362 draft tokens proposed / 2919 accepted (thousands of hipCUB TOP_K
  calls). Output scanned end-to-end (not just head/tail): zero non-printable/control characters,
  no corruption signature, coherent Python throughout, ~21 t/s.
- **Multi-turn conversations, natural EOG stops:** 16 turns total (the EOG-fix verification
  above), both `--spec-draft-n-max 3` and `8`, all with graphs enabled — clean throughout.
- **Concurrent multi-stream (`--parallel 4`), the one dimension the 2026-09-02 entry above
  didn't try:** 4-way concurrent requests across code/prose/JSON/hex prompt types, several
  rounds. No garbage token IDs, no `Invalid input batch`, no crashes — but see below.

Across everything tried this session and the 2026-09-02 session before it (single-stream up to
3800 tokens, up to ~56K context, 4-turn and 16-turn conversations, `--spec-draft-n-max` 3 through
8, and now 4-way concurrency) — **zero reproductions of the originally-reported garbage-token-IDs
signature, with graphs enabled throughout.** This still isn't a disproof (a rarer trigger than
anything tried remains possible), but the evidence has now accumulated across two sessions and
a materially wider test matrix without a single hit, against a real, measured ~30–35% throughput
cost for keeping `GGML_CUDA_DISABLE_GRAPHS=1` set. Recommendation: worth an operator-supervised
trial of removing the flag from a production config (e.g. `qwen38-27b`, already MTP-enabled) with
normal traffic and close output monitoring for a period, rather than either keeping the flag
forever on the strength of one now-unreproducible 2026-09-01 finding, or removing it unilaterally
from this session.

**A different, real, reproducible bug was found instead, while probing the concurrency
dimension** — content cross-contamination between concurrent MTP slots. With 4 concurrent
requests carrying clearly distinct prompts (e.g. quicksort code, a Matrix class, "the Roman
Empire...", "the CAP theorem..."), responses would drift into content that only makes sense in
a *different* slot's request — e.g. a quicksort-completion response drifting into "Caesar's
assassination in 44 BC", or a CAP-theorem response drifting into `def __arr) + right\n    return
quick_sort(arr)`. Output stays valid UTF-8 with no garbage-token signature, which is arguably
worse: it reads as plausible, on-topic-adjacent text rather than obviously broken. **Confirmed to
reproduce identically with `GGML_CUDA_DISABLE_GRAPHS=1` set** (control run, same prompts, same
`--parallel 4`) — so this is unrelated to HIP graph capture, a separate, apparently undocumented
`--parallel` + MTP interaction bug, most likely in how PR #27836's still-draft code handles the
draft model's own per-sequence state across concurrent slots. Ruled out as a client-side
test-harness artifact: requests use `ThreadPoolExecutor.submit(call, i, p)` (arguments bound at
submit time, not late-bound), and a control test with maximally-repetitive prompts (e.g. "write
ROCKET forty times" vs. "write BANANA forty times" concurrently) showed zero contamination across
3 rounds — the effect is real but needs enough content complexity to surface, which is also why a
casual smoke test wouldn't have caught it.

**Does not affect Tohil's production catalog today** — every MTP-using config there
(`qwen36-35b-a3b`, `qwen38-27b`/`-rocm`/`-vk`, both `gemma4-26b-a4b` configs) runs `--parallel 1`
exclusively (see `docs/v5-ops-sprints...` / the ai-mode catalog, not this repo). Left as a
documented, reproducible finding rather than chased into a fix — out of scope for this session
(explicitly deferred per the ai-mode CLAUDE.md: concurrent serving for qwen4exp specifically is
adjacent to the separate vLLM-integration track, and #28019's `--parallel`-only rollback gap is
already known/deliberately parked for the same reason). Worth an upstream report if/when someone
picks up concurrent MTP serving; not filed from this session.

### Resolved, same day: the real trigger was `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, not graphs alone

The "still did not reproduce" verdict above turned out to be right for the wrong reason: none of
this session's (or 2026-09-02's) standalone test harnesses ever set
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON`, and that omission is exactly why every test stayed clean.
Live-loading Qwen3.8-Flash-Next through the *real* `forge load` path (not a standalone bypass)
crashed on the very first request — a hard `HSA_STATUS_ERROR_MEMORY_FAULT` page fault inside
`k_get_rows_float`, the token-embedding lookup kernel. Isolated with a clean A/B/C, live on
Tohil: known-good config (`-ngl 999 -ngld 999`, graphs on, no unified-memory env var) → clean.
Same config plus `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON` → crash on the first request, every time.
Same config plus the env var plus `GGML_CUDA_DISABLE_GRAPHS=1` → clean again.

The env var is real (`getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY")` in `ggml_cuda_device_malloc`,
`ggml-cuda.cu` — switches the allocator from `hipMalloc` to `hipMallocManaged`), but the
compile-time `GGML_HIP_UMA` CMake flag every ROCm build here has always passed alongside it is a
total no-op — grep the tree, it's never read by any `#if`/`#ifdef`. And the ~63 GB ceiling this
env var is documented (in the ai-mode CLAUDE.md) to lift didn't reproduce either: a 90 GB
Flash-Next load at full 262144 context worked fine on plain `hipMalloc` alone, most likely
because this is a true APU with no separate VRAM — the kernel's own GTT mechanism already
provides broad system-RAM access regardless of the userspace allocator.

**Fix:** removed `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON` from all four of Tohil's slot env files
(`/etc/sysconfig/forge-a{1,2,3,4}-env`) — no `forge` Go code change needed, since
`writeServiceFiles` only *preserves* whatever's already there for non-`FORGE_` keys, never
writes this one fresh. Re-verified clean through the real production path afterward, twice (the
exact sequence that crashed before, then an 8-turn soak), full throughput, correct cross-turn
cache reuse, host healthy throughout. `GGML_CUDA_DISABLE_GRAPHS=1` should not be needed on any
ROCm+MTP config on this host anymore. Full narrative: `progress.md`'s 2026-09-03
"GGML_CUDA_ENABLE_UNIFIED_MEMORY was the actual culprit" entry; the ai-mode `CLAUDE.md` "Unified
Memory" bullet also carries the correction. Not yet re-verified above ~90 GB or on a
non-hybrid/non-MTP ROCm mode.

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
