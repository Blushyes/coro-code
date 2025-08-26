# 🚀 Coro Code

<div align="center">

**语言:** [English](README.md) | [中文](README_zh.md)

_用 Rust 编写的高性能 AI 编码代理，带有丰富的终端界面_

![demo](./images/demo.gif)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

</div>

---

Coro Code 是一个用 Rust 编写的高性能 AI 编码代理，带有丰富的终端界面。此前名为 **Trae Agent Rust**，现已更名并聚焦于速度、稳定性与优秀的使用体验，同时保持与原始工具规范的兼容性。

## ✨ 亮点

- 🦀 **纯 Rust 内核**与简洁清晰的架构，输出层抽象良好
- 🎨 **基于 iocraft 的交互式终端 UI**，实时状态与动画
- 🛠️ **强大的工具系统**：bash、edit、json_edit、thinking、task_done、ckg、mcp
- 🤖 **模型提供商**：已支持 OpenAI；Anthropic 与 Google 即将到来
- 🔍 **智能文件搜索**：@path 语法、感知 Git、极速匹配

## 🚀 快速开始

### 📋 前置条件

- 🦀 Rust 稳定版（1.70+）
- 🔑 模型 API Key（推荐 OpenAI；Anthropic/Google 即将支持）

### 📦 安装

```bash
cargo install --git https://github.com/Blushyes/coro-code --bin coro
```

### ▶️ 运行

```bash
# 交互模式（推荐）
coro

# 单次任务
coro "Fix the bug in main.rs"
```

## ⚙️ 最简配置

可用环境变量或 JSON 文件进行配置。

**方案 A：** 环境变量

```bash
# OpenAI
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o"

# 可选：为 OpenAI 兼容 API 设置自定义 base URL 和模型
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"

# 或使用通用覆盖（适用于任何协议）
export CORO_BASE_URL="https://api.custom.com"
export CORO_MODEL="custom-model"
```

**方案 B：** 工作目录中的 JSON 文件（`coro.json`）

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

### 🤖 支持的模型

| 提供商           | 模型                    | 状态      |
| ---------------- | ----------------------- | --------- |
| 🟢 **OpenAI**    | `gpt-4o`, `gpt-4o-mini` | ✅ 已支持 |
| 🟡 **Anthropic** | `claude-3.5` 系列       | 🚧 计划中 |
| 🔵 **Google**    | `gemini-1.5` 系列       | 🚧 计划中 |

### 🔧 环境变量参考

| 变量名                  | 描述                                 | 示例                                        |
| ----------------------- | ------------------------------------ | ------------------------------------------- |
| `OPENAI_API_KEY`        | OpenAI API 密钥                      | `sk-...`                                    |
| `OPENAI_BASE_URL`       | OpenAI 兼容 API 的自定义 base URL    | `https://api.deepseek.com`                  |
| `OPENAI_MODEL`          | OpenAI 兼容 API 的自定义模型         | `gpt-4o`, `deepseek-chat`                   |
| `ANTHROPIC_API_KEY`     | Anthropic API 密钥                   | `sk-ant-...`                                |
| `ANTHROPIC_BASE_URL`    | Anthropic API 的自定义 base URL      | `https://api.anthropic.com`                 |
| `ANTHROPIC_MODEL`       | Anthropic API 的自定义模型           | `claude-3-5-sonnet-20241022`                |
| `GOOGLE_API_KEY`        | Google AI API 密钥                   | `AIza...`                                   |
| `GOOGLE_BASE_URL`       | Google AI API 的自定义 base URL      | `https://generativelanguage.googleapis.com` |
| `GOOGLE_MODEL`          | Google AI API 的自定义模型           | `gemini-pro`, `gemini-1.5-pro`              |
| `AZURE_OPENAI_API_KEY`  | Azure OpenAI API 密钥                | `...`                                       |
| `AZURE_OPENAI_BASE_URL` | Azure OpenAI 端点                    | `https://your-resource.openai.azure.com`    |
| `AZURE_OPENAI_MODEL`    | Azure OpenAI 的自定义模型            | `gpt-4`, `gpt-35-turbo`                     |
| `CORO_BASE_URL`         | 通用 base URL 覆盖（适用于任何协议） | `https://api.custom.com`                    |
| `CORO_PROTOCOL`         | 强制指定协议                         | `openai`, `anthropic`                       |
| `CORO_MODEL`            | 通用模型覆盖（适用于任何协议）       | `gpt-4o`, `claude-3-5-sonnet`               |

## 🗺️ 开发路线图

**状态说明：** ✅ 已完成 | 🚧 进行中 | 📋 计划中

<details>
<summary><strong>🚀 第一阶段：核心体验</strong></summary>

| 优先级 | 状态 | 功能特性                   | 描述                                                                              |
| ------ | ---- | -------------------------- | --------------------------------------------------------------------------------- |
| 🔥 高  | 🚧   | **首次进入配置管理**       | 引导式向导（检测/创建 openai.json 或环境变量），校验 API Key，提供默认模型与示例  |
| 🔥 高  | ✅   | **重构、优化配置加载逻辑** | 统一优先级（CLI 参数 > 环境变量 > JSON 文件）、更友好的错误提示与诊断、可选热加载 |
| 🔥 高  | 📋   | **Tool Call 权限系统**     | 按工具/命令/目录白名单、交互确认、防越权与敏感操作提示                            |

</details>

<details>
<summary><strong>🎨 第二阶段：用户体验增强</strong></summary>

| 优先级 | 状态 | 功能特性                                  | 描述                                                   |
| ------ | ---- | ----------------------------------------- | ------------------------------------------------------ |
| 🟡 中  | 📋   | **支持 CORO.md 自定义提示词**             | 项目/子目录级覆盖、场景化模板（bugfix/重构/文档/测试） |
| 🟡 中  | 🚧   | **UI 布局优化与统一化**                   | Header/Status/Input 风格统一、键位与交互一致性优化     |
| 🟡 中  | 📋   | **轨迹回放与导出**                        | Trajectory 可视化、一键回放、导出为 JSON/Markdown      |
| 🎨 低  | 📋   | **需要一个和 gemini-cli 风格类似的 logo** | 视觉标识设计                                           |

</details>

<details>
<summary><strong>🤖 第三阶段：智能化与性能</strong></summary>

| 优先级 | 状态 | 功能特性             | 描述                                                |
| ------ | ---- | -------------------- | --------------------------------------------------- |
| 🟡 中  | 📋   | **多模型与自动路由** | 按任务类型自动选择模型，失败自动降级与重试策略      |
| 🟡 中  | 📋   | **上下文优化与缓存** | 文件摘要缓存、重复引用去重、Token 预算控制          |
| 🟡 中  | ✅   | **Token 压缩**       | 智能上下文压缩、选择性 Token 减少、自适应上下文窗口 |
| 🔵 低  | 📋   | **MCP 扩展生态**     | 常用 Provider 预设与模板，一键启停外部工具          |

</details>

<details>
<summary><strong>🌐 第四阶段：平台与生态</strong></summary>

| 优先级 | 状态 | 功能特性                 | 描述                                          |
| ------ | ---- | ------------------------ | --------------------------------------------- |
| 🔵 低  | 📋   | **core 支持打包为 WASM** | 浏览器/插件环境可用，同构工具接口与最小运行时 |
| 🔵 低  | 📋   | **跨平台增强**           | macOS/Linux/Windows/WSL 细节适配与稳定性提升  |
| 🔵 低  | 📋   | **插件化工具系统**       | 第三方工具注册规范、版本与依赖声明            |

</details>

<details>
<summary><strong>🛡️ 第五阶段：安全与质量</strong></summary>

| 优先级 | 状态 | 功能特性           | 描述                                           |
| ------ | ---- | ------------------ | ---------------------------------------------- |
| 🟡 中  | 📋   | **安全与速率限制** | 沙箱模式（受限 bash/网络开关）、并发与速率限制 |
| 🔵 低  | 📋   | **测试与基准**     | 端到端测试样例、性能基准与对比报告             |

</details>

## 🛠️ 开发

### Pre-commit Hooks

我们强烈建议设置 pre-commit hooks 来维护代码质量。仓库包含了自动安装 hooks 的脚本，这些 hooks 会在每次提交前运行格式化、代码检查和测试。

根据你的平台选择合适的脚本：

```bash
# Linux/macOS
./scripts/setup-pre-commit-hooks.sh

# Windows PowerShell
.\scripts\setup-pre-commit-hooks.ps1

# Windows 命令提示符
scripts\setup-pre-commit-hooks.bat
```

Pre-commit hook 会自动运行：

- **代码格式化** (`cargo fmt --check`)
- **代码检查** (`cargo clippy`)
- **测试** (`cargo test`)

更多详情请参见 [scripts/README.md](scripts/README.md)。

### 贡献代码

1. Fork 仓库
2. 创建功能分支
3. **设置 pre-commit hooks**（推荐）
4. 进行修改
5. 确保所有测试通过
6. 提交 Pull Request

## 📄 许可证

双许可证，任选其一：

- **Apache-2.0** ([LICENSE-APACHE](LICENSE-APACHE))
- **MIT** ([LICENSE-MIT](LICENSE-MIT))

## 🙏 致谢

- **[Trae Agent](https://github.com/trae-ai/trae-agent)** 原始 Python 实现与规范
- **[iocraft](https://github.com/ccbrown/iocraft)** 优秀的终端 UI 框架
- **OpenAI、Anthropic、Google** 模型与 API
- **Rust 社区** 出色的生态与工具

---

<div align="center">

用 ❤️ 和 Rust 制作

</div>
