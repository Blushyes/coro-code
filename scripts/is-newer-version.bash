#!/usr/bin/env bash
# 判断是否需要发行新版本 | Checking for a Newer Version

# 设置定量 | Quantities
## 当前脚本所在目录 | Current Script Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
## 仓库目录 | Repository Directory
REPO_DIR="$(dirname "$SCRIPT_DIR")"
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

# 获取当前版本号 | Get current version number
CURRENT_VERSION=$("$SCRIPT_DIR/get-version.bash")
if [ $? -ne 0 ]; then
  exit 1
fi

# 获取最新的 git 标签版本 | Get the latest git tag version
LATEST_TAG=$(git -C "$REPO_DIR" tag -l 'v*' --sort=-v:refname | head -1)
if [ -z "$LATEST_TAG" ]; then
  # 没有标签时默认为需要发布 | No tags, assume it needs to be released
  echo "$CURRENT_VERSION"
  exit 0
fi

# 去除标签中的 v 前缀 | Remove the v prefix from the tag
LATEST_VERSION=${LATEST_TAG#v}

# 比较版本号 | Compare version numbers
if [ "$(printf "%s\n%s" "$CURRENT_VERSION" "$LATEST_VERSION" | sort -V | head -n1)" != "$CURRENT_VERSION" ]; then
  # 当前版本较新，需要发布 | The current version is newer, needs to be released
  echo "$CURRENT_VERSION"
else
  # 已经是最新版本 | Already the latest version
  echo "0"
fi
