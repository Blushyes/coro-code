#!/usr/bin/env bash
# 从 Cargo.toml 中获取版本号 | Get version number from Cargo.toml

# 设置定量 | Quantities
## 当前脚本所在目录 | Current Script Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
## 仓库目录 | Repository Directory
REPO_DIR="$(dirname "$SCRIPT_DIR")"
## 文件路径 | File Path
if [ -n "$1" ]; then
  FILE_PATH="$1"
else
  FILE_PATH="$REPO_DIR/Cargo.toml"
fi
## 当前语言 | Current Language
CURRENT_LANG=0 ### 0: en-US, 1: zh-Hans-CN

# 语言检测 | Language Detection
if [ $(echo ${LANG/_/-} | grep -Ei "\\b(zh|cn)\\b") ]; then CURRENT_LANG=1;  fi

# 本地化 | Localization
recho() {
  if [ "$CURRENT_LANG" == "1" ]; then
    ## zh-Hans-CN
    echo "$1";
  else
    ## en-US
    echo "$2";
  fi
}

# 提取版本号并打印 | Extract version number and print
if [ ! -f "$FILE_PATH" ]; then
  recho "错误：文件 $FILE_PATH 不存在" "Error: File $FILE_PATH does not exist"
  exit 1
fi

## 从 Cargo.toml 中提取 [workspace.package] 下 version 的值
VERSION=$(grep -A 10 "\[workspace\.package\]" "$FILE_PATH" | grep "^version" | sed -E 's/version = "([^"]+)"/\1/')
if [ -z "$VERSION" ]; then
  recho "错误：未找到版本号" "Error: Version not found"
  exit 1
fi

echo "$VERSION"
