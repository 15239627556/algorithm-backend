#!/bin/bash
#
# Triton 启动前自动选择 plan 文件
# 根据 CUDA_VISIBLE_DEVICES (UUID) 检测 GPU 型号，将 model_<tag>.plan 软链接为 model.plan
#
# 必须由 docker-compose 的 entrypoint 调用本脚本，否则仅挂载文件不会执行，软链永远不会更新。
#
# 优先级:
#   1. 环境变量 GPU_TYPE (手动指定，如 GPU_TYPE=3080)
#   2. 通过 CUDA_VISIBLE_DEVICES 中的 UUID 从 nvidia-smi 自动检测

MODEL_REPO=${MODEL_REPO:-/models}

# 将 UUID 规范为不含 GPU- 前缀的小写串，便于与 nvidia-smi 输出比对
normalize_uuid() {
    local u="$1"
    u=$(echo "$u" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    u="${u#gpu-}"
    echo "$u"
}

# 从 nvidia-smi 的 csv 行中按 UUID 解析 GPU 名称（兼容 "uuid,name" / "uuid, name"）
name_for_gpu_uuid() {
    local want
    want=$(normalize_uuid "$1")
    [ -z "$want" ] && return 1

    local line u n u_norm
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        u="${line%%,*}"
        n="${line#*,}"
        u=$(echo "$u" | tr -d '[:space:]')
        n=$(echo "$n" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        u_norm=$(normalize_uuid "$u")
        if [ "$u_norm" = "$want" ]; then
            echo "$n"
            return 0
        fi
    done < <(nvidia-smi --query-gpu=gpu_uuid,name --format=csv,noheader 2>/dev/null)
    return 1
}

detect_gpu_tag() {
    local cuda_dev="$CUDA_VISIBLE_DEVICES"
    local gpu_name=""

    # 如果是 UUID 格式 (GPU-xxxx)，通过 nvidia-smi 反查名称
    if echo "$cuda_dev" | grep -qi "^GPU-"; then
        local uuid="${cuda_dev%%,*}"
        if gpu_name=$(name_for_gpu_uuid "$uuid"); then
            echo "通过 UUID $uuid 检测到: $gpu_name" >&2
        else
            # 容器内常仅暴露一张卡，此时用可见设备 0 即当前 CUDA_VISIBLE_DEVICES 对应卡
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            echo "[select_plan] UUID 列表匹配失败，改用可见 GPU 0: $gpu_name" >&2
        fi
    else
        # 数字编号 fallback
        local gpu_id="${cuda_dev:-0}"
        gpu_id="${gpu_id%%,*}"
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader --id="$gpu_id" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        echo "通过设备 $gpu_id 检测到: $gpu_name" >&2
    fi

    if [ -z "$gpu_name" ]; then
        echo "[select_plan] 无法根据 CUDA_VISIBLE_DEVICES=$cuda_dev 解析 GPU 名称" >&2
        echo "[select_plan] 当前可见 GPU 列表:" >&2
        nvidia-smi --query-gpu=gpu_uuid,name --format=csv,noheader 2>/dev/null >&2 || true
        echo ""
        return
    fi

    for tag in 4090 4080 4070 4060 3090 3080 3070 3060 2080 2070 2060 A100 A10 H100 H200 L40 L4; do
        if echo "$gpu_name" | grep -qi "$tag"; then
            echo "$tag"
            return
        fi
    done

    echo ""
}

# 确定 GPU 标签
if [ -n "$GPU_TYPE" ]; then
    GPU_TAG="$GPU_TYPE"
    echo "[select_plan] 使用环境变量 GPU_TYPE=$GPU_TAG"
else
    GPU_TAG=$(detect_gpu_tag)
    if [ -z "$GPU_TAG" ]; then
        echo "[select_plan] 警告: 无法识别 GPU 型号，跳过 plan 选择，使用已有的 model.plan"
        exec "$@"
    fi
    echo "[select_plan] 自动检测 GPU_TAG=$GPU_TAG"
fi

echo "[select_plan] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 遍历所有模型目录，创建软链接
linked=0
skipped=0

for version_dir in "$MODEL_REPO"/*/1/; do
    [ -d "$version_dir" ] || continue

    target="$version_dir/model_${GPU_TAG}.plan"
    link="$version_dir/model.plan"
    model_name=$(basename "$(dirname "$version_dir")")

    if [ -f "$target" ]; then
        rm -f "$link"
        ln -s "model_${GPU_TAG}.plan" "$link"
        echo "  [OK] $model_name -> model_${GPU_TAG}.plan"
        linked=$((linked + 1))
    else
        skipped=$((skipped + 1))
    fi
done

echo "[select_plan] 完成: 链接 $linked 个模型, 跳过 $skipped 个 (无对应 plan)"
echo "========================================"

exec "$@"
