# 🚀 Coro Code

<div align="center">

**Language:** [English](README.md) | [中文](README_zh.md)

_A high-performance AI coding agent written in Rust with a rich terminal UI_

![demo](./images/demo.gif)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

</div>

---

Coro Code is a high-performance AI coding agent written in Rust with a rich terminal UI. Formerly known as **Trae Agent Rust**, it remains compatible with the original tool spec while focusing on speed, reliability, and great UX.

## ✨ Highlights

- 🚀 **High Performance**: Written in Rust for speed and memory safety
- 🎨 **Rich Terminal UI**: Beautiful, interactive interface with real-time updates
- 🔧 **Easy Configuration**: Support for multiple LLM providers with flexible config options
- 🛠️ **Powerful Tools**: Built-in bash execution, file operations, and extensible tool system
- 🔄 **Environment Variables**: Comprehensive support for API keys, base URLs, and model configuration
- 📦 **Cross-Platform**: Works seamlessly on macOS, Linux, and Windows

## 🚀 Quick Start

### 📋 Prerequisites

- 🦀 Rust stable (1.70+)
- 🔑 An API key (OpenAI recommended; Anthropic/Google coming soon)

### 📦 Install

```bash
cargo install --git https://github.com/Blushyes/coro-code --bin coro
```

### ▶️ Run

```bash
# Interactive mode (recommended)
coro

# Single task
coro "Fix the bug in main.rs"
```

### Configuration

**Option A:** Environment variables

```bash
# OpenAI
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o"

# Optional: Custom base URL and model for OpenAI-compatible APIs
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"

# Or use generic overrides for any protocol
export CORO_BASE_URL="https://api.custom.com"
export CORO_MODEL="custom-model"
```

**Option B:** Configuration file

Create a `coro.json` file:

```json
{
  "protocol": "openai",
  "base_url": "https://api.deepseek.com",
  "api_key": "your-api-key",
  "model": "deepseek-chat",
  "params": {
    "max_tokens": 131072,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

### Usage

```bash
# Interactive mode
coro

# Direct command
coro "Help me refactor this function"

# With specific config
coro --config custom.json "Analyze this codebase"
```

## 🤖 Supported Models

| Provider         | Models                  | Status    |
| ---------------- | ----------------------- | --------- |
| 🟢 **OpenAI**    | `gpt-4o`, `gpt-4o-mini` | ✅ Ready  |
| 🟡 **Anthropic** | `claude-3.5` family     | 🚧 Coming |
| 🔵 **Google**    | `gemini-1.5` family     | 🚧 Coming |

### 🔧 Environment Variables Reference

| Variable                | Description                                | Example                                     |
| ----------------------- | ------------------------------------------ | ------------------------------------------- |
| `OPENAI_API_KEY`        | OpenAI API key                             | `sk-...`                                    |
| `OPENAI_BASE_URL`       | Custom base URL for OpenAI-compatible APIs | `https://api.deepseek.com`                  |
| `OPENAI_MODEL`          | Custom model for OpenAI-compatible APIs    | `gpt-4o`, `deepseek-chat`                   |
| `ANTHROPIC_API_KEY`     | Anthropic API key                          | `sk-ant-...`                                |
| `ANTHROPIC_BASE_URL`    | Custom base URL for Anthropic API          | `https://api.anthropic.com`                 |
| `ANTHROPIC_MODEL`       | Custom model for Anthropic API             | `claude-3-5-sonnet-20241022`                |
| `GOOGLE_API_KEY`        | Google AI API key                          | `AIza...`                                   |
| `GOOGLE_BASE_URL`       | Custom base URL for Google AI API          | `https://generativelanguage.googleapis.com` |
| `GOOGLE_MODEL`          | Custom model for Google AI API             | `gemini-pro`, `gemini-1.5-pro`              |
| `AZURE_OPENAI_API_KEY`  | Azure OpenAI API key                       | `...`                                       |
| `AZURE_OPENAI_BASE_URL` | Azure OpenAI endpoint                      | `https://your-resource.openai.azure.com`    |
| `AZURE_OPENAI_MODEL`    | Custom model for Azure OpenAI              | `gpt-4`, `gpt-35-turbo`                     |
| `CORO_BASE_URL`         | Generic base URL override (any protocol)   | `https://api.custom.com`                    |
| `CORO_PROTOCOL`         | Force specific protocol                    | `openai`, `anthropic`                       |
| `CORO_MODEL`            | Generic model override (any protocol)      | `gpt-4o`, `claude-3-5-sonnet`               |

## 🗺️ Roadmap

**Status Legend:** ✅ Completed | 🚧 In Progress | 📋 Planned

<details>
<summary><strong>🚀 Phase 1: Core Experience</strong></summary>

| Priority | Status | Feature                           | Description                                                                                                     |
| -------- | ------ | --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 🔥 High  | 🚧     | **First-time Setup Management**   | Guided wizard (detect/create openai.json or env vars), API key validation, default models & examples            |
| 🔥 High  | ✅     | **Refactor Config Loading Logic** | Unified priority (CLI args > env vars > JSON file), friendly error messages & diagnostics, optional hot reload  |
| 🔥 High  | 📋     | **Tool Call Permission System**   | Tool/command/directory whitelist, interactive confirmation, privilege escalation & sensitive operation warnings |

</details>

<details>
<summary><strong>🎨 Phase 2: User Experience Enhancement</strong></summary>

| Priority  | Status | Feature                                  | Description                                                                                      |
| --------- | ------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 🟡 Medium | 📋     | **CORO.md Custom Prompts Support**       | Project/subdirectory level overrides, scenario templates (bugfix/refactor/docs/test)             |
| 🟡 Medium | 🚧     | **UI Layout Optimization & Unification** | Header/Status/Input style consistency, keyboard shortcuts & interaction consistency optimization |
| 🟡 Medium | 📋     | **Trajectory Replay & Export**           | Trajectory visualization, one-click replay, export to JSON/Markdown                              |
| 🎨 Low    | 📋     | **Logo Design (gemini-cli style)**       | Visual identity design                                                                           |

</details>

<details>
<summary><strong>🤖 Phase 3: Intelligence & Performance</strong></summary>

| Priority  | Status | Feature                            | Description                                                                          |
| --------- | ------ | ---------------------------------- | ------------------------------------------------------------------------------------ |
| 🟡 Medium | 📋     | **Multi-model & Auto Routing**     | Auto model selection by task type, failure auto-downgrade & retry strategies         |
| 🟡 Medium | 📋     | **Context Optimization & Caching** | File summary caching, duplicate reference deduplication, token budget control        |
| 🟡 Medium | ✅     | **Token Compression**              | Intelligent context compression, selective token reduction, adaptive context windows |
| 🔵 Low    | 📋     | **MCP Extension Ecosystem**        | Common provider presets & templates, one-click start/stop external tools             |

</details>

<details>
<summary><strong>🌐 Phase 4: Platform & Ecosystem</strong></summary>

| Priority | Status | Feature                        | Description                                                                   |
| -------- | ------ | ------------------------------ | ----------------------------------------------------------------------------- |
| 🔵 Low   | 📋     | **Core WASM Support**          | Browser/plugin environment ready, isomorphic tool interface & minimal runtime |
| 🔵 Low   | 📋     | **Cross-platform Enhancement** | macOS/Linux/Windows/WSL detail adaptation & stability improvements            |
| 🔵 Low   | 📋     | **Plugin Tool System**         | Third-party tool registration spec, version & dependency declaration          |

</details>

<details>
<summary><strong>🛡️ Phase 5: Security & Quality</strong></summary>

| Priority  | Status | Feature                      | Description                                                                  |
| --------- | ------ | ---------------------------- | ---------------------------------------------------------------------------- |
| 🟡 Medium | 📋     | **Security & Rate Limiting** | Sandbox mode (restricted bash/network switches), concurrency & rate limiting |
| 🔵 Low    | 📋     | **Testing & Benchmarks**     | End-to-end test cases, performance benchmarks & comparison reports           |

</details>

## 🛠️ Development

### Pre-commit Hooks

We strongly recommend setting up pre-commit hooks to maintain code quality. The repository includes scripts to automatically install hooks that run formatting, linting, and tests before each commit.

Choose the appropriate script for your platform:

```bash
# Linux/macOS
./scripts/setup-pre-commit-hooks.sh

# Windows PowerShell
.\scripts\setup-pre-commit-hooks.ps1

# Windows Command Prompt
scripts\setup-pre-commit-hooks.bat
```

The pre-commit hook will automatically run:

- **Code formatting** (`cargo fmt --check`)
- **Linting** (`cargo clippy`)
- **Tests** (`cargo test`)

For more details, see [scripts/README.md](scripts/README.md).

### Contributing

1. Fork the repository
2. Create a feature branch
3. **Set up pre-commit hooks** (recommended)
4. Make your changes
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

Dual licensed under your choice of:

- **Apache-2.0** ([LICENSE-APACHE](LICENSE-APACHE))
- **MIT** ([LICENSE-MIT](LICENSE-MIT))

## 🙏 Acknowledgments

- **[Trae Agent](https://github.com/bytedance/trae-agent)** for the original Python implementation and spec
- **[iocraft](https://github.com/ccbrown/iocraft)** for the beautiful terminal UI framework
- **OpenAI, Anthropic, and Google** for model APIs
- **Rust community** for the amazing ecosystem

---

<div align="center">

Made with ❤️ in Rust

</div>
