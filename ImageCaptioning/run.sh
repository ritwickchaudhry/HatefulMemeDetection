INPUT='/home/ubuntu/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl'
OUTPUT='/home/ubuntu/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/dev_seen_with_captions.jsonl'
IMAGE_DIR='/home/ubuntu/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images'
python -W ignore generate_captions.py --input_file $INPUT --output_file $OUTPUT --image_dir $IMAGE_DIR --model=BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map='WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5
