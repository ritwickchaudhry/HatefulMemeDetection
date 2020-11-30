mmf_run config=projects/hateful_memes/configs/visual_bert/defaults.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=val \
    checkpoint.resume_file=/home/ubuntu/HatefulMemeDetection/VisualBERTLogs/best.ckpt \
    checkpoint.resume_pretrained=False
