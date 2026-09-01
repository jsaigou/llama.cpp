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

**Known requirement — ROCm-TheRock builds:** on Tohil's ROCm-TheRock 10.1 toolchain,
the hipCUB TOP_K path hits graph-capture corruption (garbage decoded token IDs) unless
launched with `GGML_CUDA_DISABLE_GRAPHS=1`. This is the same class of issue flagged in
PR #27836's review thread (missing rocPRIM capture guard, upstream-fixed in ROCm 7.13+)
— just not yet confirmed fixed on the TheRock variant. Standard ROCm 7.1/7.2.3 builds
per community reports in the PR thread do not need this workaround.

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
