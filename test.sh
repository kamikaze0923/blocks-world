#!/bin/bash

echo "this test will just render the first 10 images of 2 objs, 2 stacks blocksworld."
read -n 1 -s -r -p "Press any key to continue"

blender -noaudio --background --python render_images.py -- \
      --output-dir      output                          \
      --initial-objects output/output-init.json                \
      --statistics      output/output-stat.json                \
      --render-num-samples 20                         \
      --start-idx          0                            \
      --width 150                                       \
      --height 100                                      \
      --num-objects 3                                  \
      --max-stacks  3                                   \
#      --dry-run


# ./extract_region.py output/scenes/CLEVR_new_000000_pre.json
# ./extract_region.py output/scenes/CLEVR_new_000000_suc.json
