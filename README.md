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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## This fork (jsaigou/llama.cpp)

A working fork of upstream `ggml-org/llama.cpp`, maintained for **Tohil**, a self-hosted
inference host (AMD Strix Halo, Ryzen AI Max+ 395, `gfx1151`, 128 GB unified memory).
**Branch `kintsugi`** (merged 2026-10-23; tree identical to `rebase/upstream-20261022`) —
upstream `master` at `7170cad4d` (2026-10-22) plus two patches (below). The pre-rebase history
(based on upstream `458681e1d`, 2026-09-01) is the merge's first parent, so
`git show edfb0e840:README.md` has the long-form notes.

### Patches carried on top of upstream

1. **Kintsugi — cross-turn KV cache reuse for hybrid/recurrent architectures**
   (`tools/server/server-context.cpp`, `common/common.{h,cpp}`). Lets hybrid/recurrent
   models (Qwen3.5/3.6 MoE, qwen4exp, ...) reuse cache across conversation turns instead of
   re-prefilling the whole conversation: a dedicated generation-end checkpoint per turn,
   preferred on restore, a GPU sync after partial-state restore, and checkpoints preserved
   across a forced position reset. Production, stable.
2. **MTP checkpoint loss past EOG** ([ggml-org/llama.cpp#28049](https://github.com/ggml-org/llama.cpp/issues/28049)).
   A speculative-decode round that ended generation never saved a Kintsugi checkpoint, so
   the next turn re-prefilled the entire answer just generated. Not specific to one model:
   it hits any hybrid/recurrent model served with `--spec-type draft-mtp`. The fix shares one
   `save_generation_checkpoint()` between the plain and speculative paths and forces the
   existing restore-and-replay path when an EOG lands mid-batch.

### What changed in the 2026-10-22 rebase

- **The qwen4exp MTP patch set was dropped, not ported.** Upstream merged qwen4exp MTP
  itself (#29761), so the 11 earlier community-PR commits (NextN head, converter, the
  detached-head loader, the follow-up correctness fixes) are obsolete. The detached MTP
  head GGUF (`mtp-*.gguf` via `--spec-draft-model`) loads unchanged on upstream.
- **`--no-mmap` no longer exists.** Upstream removed `--mmap`/`--no-mmap`/`--mlock`/
  `--direct-io` in favor of `--load-mode {auto,none,mmap,mlock,mmap+mlock}` (#28334).
  `--no-mmap` becomes `--load-mode none`. Builds from September 2026 onward already accept
  both spellings; builds from this rebase reject the old one.
- **`preserve_reasoning` is on by default** for templates that support it (`--no-reasoning-preserve` to turn off).

### Verification (Tohil, 2026-10-22/23, Qwen3.8-Flash-Next UD-Q3_K_XL + Q8_0 MTP head)

ROCm build (TheRock 10.1, gfx1151), 262144 context, `--spec-type draft-mtp
--spec-draft-n-max 3`, q8_0 KV, vision projector attached, 8 long multi-turn exchanges:

| | previous `kintsugi` build | this branch |
|---|---|---|
| cache reuse by turn 8 (`cache_n`) | ~5,070 | ~5,070 |
| new prompt tokens per turn | 24-35 | 24-35 |
| decode | 16-18 t/s | 30-37 t/s |
| image question | correct | correct |

No aborts, faults, or checkpoint-restore loops. Decode is roughly doubled; the cause was
not isolated (upstream's ROCm and qwen4exp kernel work since September is the likely source).

### Hardware notes that outlived the old README

- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON` is **not** needed on this true-APU host and, combined
  with HIP graph capture (any `--spec-type draft-mtp` config), causes a hard
  `HSA_STATUS_ERROR_MEMORY_FAULT` in `k_get_rows_float`. It — not HIP graphs alone — was the
  real trigger behind the old `GGML_CUDA_DISABLE_GRAPHS=1` requirement, which is therefore
  unnecessary. `GGML_HIP_UMA` is a no-op CMake variable. A 90 GB model at full 262144 context
  loads on plain `hipMalloc`.
- Run `--parallel 1` for anything serving one long conversation at full context: `--parallel N`
  *splits* the configured context across N slots.

The full original write-up (diagnosis history, the 231K-token depth soak, the EOG-fix
analysis) is in this README at commit `edfb0e840` (the previous `kintsugi` tip).

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

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
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
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
