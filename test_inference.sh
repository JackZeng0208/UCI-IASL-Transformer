#!/bin/bash
python3 test/test_inference.py \
    --model_size="125M" \
    --prompt="Today is a beautiful day, and" \
    --ip=http://128.195.54.163 \
    --port=5000