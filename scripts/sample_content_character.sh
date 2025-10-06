for ch in {A..Z}; do
    python sample.py \
        --ckpt_dir="ckpt/" \
        --style_image_path="data_examples/sampling/example_style.jpg" \
        --save_image \
        --character_input \
        --content_character="$ch" \
        --save_image_dir="outputs/" \
        --device="cpu" \
        --ttf_path="fonts/NotoSans-VariableFont_wdth,wght.ttf" \
        --algorithm_type="dpmsolver++" \
        --guidance_type="classifier-free" \
        --guidance_scale=7.5 \
        --num_inference_steps=20 \
        --method="multistep"
done
