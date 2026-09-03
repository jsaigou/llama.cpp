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

**Root cause found and fixed, 2026-09-03 — `GGML_CUDA_DISABLE_GRAPHS=1` is not actually
needed; the real problem was `GGML_CUDA_ENABLE_UNIFIED_MEMORY`.** The original 2026-09-01
finding below is kept verbatim for the record, but its diagnosis was wrong: the corruption
isn't caused by HIP graph capture alone. It's caused by `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON`
(a runtime env var, checked via `getenv()` in `ggml_cuda_device_malloc`, `ggml-cuda.cu` —
switches the allocator from `hipMalloc` to `hipMallocManaged`) combined with HIP graph
capture. Confirmed with a clean A/B/C isolation on Tohil, live: `-ngl 999 -ngld 999`, no
unified-memory env var, graphs on → clean across 24+ turns. Same config plus the env var,
graphs on → a hard `HSA_STATUS_ERROR_MEMORY_FAULT` page fault inside `k_get_rows_float`
(the token-embedding lookup kernel) on the very first request, every time. Same config plus
the env var plus `GGML_CUDA_DISABLE_GRAPHS=1` → clean again. The `GGML_HIP_UMA` CMake flag
every kintsugi ROCm build has always passed alongside it turns out to be a complete no-op —
grep the tree, it's never referenced by any `#if`/`#ifdef`, only sets a CMake cache variable.
And the documented ~63 GB ceiling this env var is supposed to lift didn't reproduce either: a
90 GB Flash-Next load at full 262144 context worked fine on plain `hipMalloc` alone, most
likely because this is a true APU with no separate VRAM — the kernel's own GTT mechanism
already gives the GPU broad system-RAM access regardless of which userspace allocator is
used. Removed from Tohil's four slot env files 2026-09-03 (`/etc/sysconfig/forge-a{1,2,3,4}-env`
— see `progress.md`'s "GGML_CUDA_ENABLE_UNIFIED_MEMORY was the actual culprit" entry and
this repo's `CLAUDE.md` "Unified Memory" bullet for the full writeup); `GGML_CUDA_DISABLE_GRAPHS=1`
should not be needed on any ROCm+MTP config on this host now that the actual trigger is gone.
Not yet re-verified above ~90 GB or on a non-hybrid/non-MTP ROCm mode.

<details>
<summary>Original 2026-09-01 finding (superseded diagnosis, kept for the record)</summary>

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

</details>

**Measured on Tohil (2026-09-01), gfx1151/ROCm, `Qwen3.8-Flash-Next-UD-Q3_K_XL` +
[`drluoto/Qwen3.8-Flash-Next-MTP-GGUF`](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF)
Q8_0 draft sidecar, 32K context, greedy decode, 400-token coding completion:**

| Config | Decode speed | vs. baseline |
|---|---|---|
| No speculation | 22.06 t/s | — |
| `--spec-type draft-mtp --spec-draft-n-max 3` | 34.5–34.7 t/s | **+56–57%** |

Draft acceptance ~86% (269–273/307–312 tokens accepted). Output verified coherent and
correct on real coding prompts.

**Not yet done:** long-context (>32K) verification on this exact model/quant, and a
production catalog promotion decision for `qwen38-flash-next` specifically. Upstream PR
#27836 is still draft/unmerged — this fork tracks it, not a stable release.

### 3. MTP checkpoint loss past EOG on hybrid models — fixed 2026-09-03

**Status: fixed and merged into `kintsugi` (source tree only, see below).** This is
[ggml-org/llama.cpp#28049](https://github.com/ggml-org/llama.cpp/issues/28049), and it is
**not specific to Qwen3.8-Flash-Next** — it affects any hybrid/recurrent model served
with `--spec-type draft-mtp`, which on Tohil's live catalog today already includes
`qwen36-35b-a3b`, `qwen38-27b`/`-rocm`/`-vk`, and both `gemma4-26b-a4b` configs (all
`visible`, all real production traffic).

Root cause: a speculative-decode accept round that ends generation (EOG, or hitting
`max_tokens`) never saved a Kintsugi generation checkpoint at all — the checkpoint
mechanism that already exists for the plain per-token sampling path was never mirrored
into the MTP accept loop. Since a hybrid/recurrent model's SSM/GDN state can't be
trimmed back to an arbitrary position (only restored from a checkpoint), the *next*
turn's prefix match fell back to whatever checkpoint was last made during the
*previous* prompt and silently re-prefilled the entire answer that had just been
generated — defeating Kintsugi's whole cross-turn cache-reuse value proposition for
exactly the multi-turn + MTP combination this fork exists to serve.

The fix (`tools/server/server-context.cpp`, see the commit for the full writeup) has
two parts: (1) extract the existing checkpoint-on-stop logic into a shared
`save_generation_checkpoint()` and call it from the MTP accept loop's stop branch too;
(2) verification-accepted draft tokens are already fully decoded into `ctx_tgt`'s
memory as one batch before per-token stop-checking even runs, so if the model's EOG
token lands anywhere in that batch except the very last decoded position, whatever
comes after it is stranded in memory and un-trimmable — a checkpoint taken naively at
that point would silently record a position past the real stop. A pre-scan for EOG
among the verified tokens forces the *existing* checkpoint-restore-and-replay path
(already used for verification-rejected drafts) to redo the round at the correct size
instead, rather than inventing a new rollback mechanism.

A real bug was found and fixed in the fix itself while verifying live on Tohil:
truncating on *any* EOG match, even one already at the last decoded position (i.e.
nothing stranded, nothing to fix), forced a pointless rollback+replay whose replay
deterministically reproduces the identical EOG token again — a genuine infinite loop,
caught live as ~2500 repeated checkpoint restores stuck at one fixed context position
and a ~9x throughput drop, before the corrected version (only roll back when a decoded
token is genuinely stranded after the EOG) was verified clean.

**Verified live on Tohil** against Qwen3.8-Flash-Next with `--spec-type draft-mtp`, both
`--spec-draft-n-max 3` and `8`: 16 total turns across two multi-turn conversations, most
ending via a real EOG stop, show `cache_n` climbing steadily turn-over-turn and
`prompt_n` staying in the low hundreds even past 2000 tokens of context — instead of
re-processing most of the prior answer, confirmed as a live control against unmodified
`kintsugi @ 189ed9bef` (turn-2 prefill ratio 0.25–0.52 of the full prior context, i.e.
re-prefilling roughly a quarter to half of everything generated so far). No
checkpoint-restore runaway, no asserts, unchanged throughput (~25–29 t/s, matching
baseline), coherent output throughout. Full detail: commit on
`fix/qwen4exp-eog-checkpoint`, merged into `kintsugi` at the source level.

**Deployed 2026-09-03, with operator sign-off.** Rebuilt `build-rocm-new` (`builds.id=10`) and
`build-vulkan-new` (`builds.id=9`) from this fix, caught a real RPATH regression in the rebuild
before it caused an outage (the original builds had `-DCMAKE_INSTALL_RPATH='$ORIGIN;...'`
set — a plain rebuild without it bakes in an absolute path that breaks the moment the build
directory gets renamed into place), fixed and reconfigured both, smoke-tested each, then
atomically swapped both into place with zero live processes running on either path (old
binaries kept as `*-pre-eogfix-20260903`, nothing deleted). Live-verified via the real
`forge load` path against Qwen3.8-Flash-Next, not just a standalone bypass — see
`progress.md`'s 2026-09-03 entries for the full narrative, including a second, more serious
bug this same verification pass surfaced (`GGML_CUDA_ENABLE_UNIFIED_MEMORY`, see this
README's `GGML_CUDA_DISABLE_GRAPHS` section above). Pushed to GitHub 2026-09-03 despite Tohil
having no cached push credentials for `jsaigou/llama.cpp`: `git bundle create` on Tohil, the
bundle transferred to a machine with a working `gh`-authenticated credential, cloned `origin`
fresh there (Tohil's own clone is shallow, so a bundle built from it can't reconstruct
standalone — cloning `origin` first and fetching the bundle's new commits on top sidesteps
that), pushed from there. `origin/kintsugi` now matches Tohil exactly (verified via
`git ls-remote`); same relay used for the other branches below.

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
