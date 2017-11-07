#!/bin/env bash

OPENPOSE_DIR=$1
TMP_DIR=/tmp
VIDEO_DIR=$2
OUTPUT_DIR=$3
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

frame_dir=${TMP_DIR}/frames
jsons_dir=${TMP_DIR}/jsons

[ -e ${frame_dir} ] || mkdir -p ${frame_dir}
[ -e ${jsons_dir} ] || mkdir -p ${jsons_dir}

cd "${OPENPOSE_DIR}"

find "${VIDEO_DIR}" -type f -name "*.M.avi" | tac | while read f; do
    batch=$(basename $(dirname "$f"))
    fold=$(basename $(dirname $(dirname "$f")))
    filename=$(echo $(basename "$f") | sed "s/\.M.avi//")
    filename="${filename%.*}"
    echo $fold/$batch/$filename
    [ ! -f "${OUTPUT_DIR}/$fold/$batch/${filename}.npy" ] || continue
    rm -f "${frame_dir}/"*
    rm -f "${jsons_dir}/"*
    </dev/null ffmpeg -loglevel warning -vsync 0 -i "${f}" -an "${frame_dir}/%010d.png"
    echo extracted $(ls "${frame_dir}"| wc -l) frames
    ./build/examples/openpose/rtpose.bin \
        -resolution 320x240 -net_resolution $((20*16))x$((15*16)) -num_scales 3 \
        --no_display -no_render_output \
        --image_dir "${frame_dir}" \
        --write_pose_json "${jsons_dir}" 1> /dev/null
    echo processed $(ls "${jsons_dir}"| wc -l) frames
    mkdir -p "${OUTPUT_DIR}/$fold/$batch"
    python "${THISDIR}/json2npy.py" \
        "${jsons_dir}" "${OUTPUT_DIR}/$fold/$batch/${filename}.npy"
done
