# DoodleAssist: Progressive Interactive Line Art Generation with Latent Distribution Alignment - TVCG 2025

[[Paper]](https://cislab.hkust-gz.edu.cn/media/documents/TVCG_DoodleAssist_final.pdf) | [[Paper (IEEE)]](https://ieeexplore.ieee.org/abstract/document/11216020) | [[Project Page]](https://markmohr.github.io/DoodleAssist/)

*DoodleAssist* is an interactive and progressive line art generation system controlled by sketches and prompts, which helps both experts and novices concretize their design intentions or explore possibilities.

<img src='docs/figures/teaser3.png'>

## Outline

- [Setup](#setup)
- [Inference](#inference)
- [Citation](#citation)

## Setup

### 1. Install Environment via Anaconda (Recommended)
```bash
conda create -n doodleassist python==3.10.15
conda activate doodleassist
pip install -r requirements.txt
```

### 2. Model Preparation

- Checkpoint

    - Download the checkpoint `controlnext-48000.bin` (13MB) [here](https://drive.google.com/file/d/1auWNemiIeChDxkUfPoGLBqdiF9AZhFM-/view?usp=sharing), and place it to `./checkpoint/controlnext-48000.bin`.


- Base model

    - We use a Stable Diffusion 1.5 model fine-tuned on line art data on civitai.com. Download it (`foolkatGODOF_v3.safetensors`) [here](https://civitai.com/models/123631?modelVersionId=142306).
    - Convert the safetensors to diffusers models using the following commands (they are placed in `./backbone/foolkatGODOF_v3/`):
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
python scripts/convert_original_stable_diffusion_to_diffusers.py \
  --checkpoint_path your/path/to/foolkatGODOF_v3.safetensors \
  --dump_path your/path/to/DoodleAssist/backbone/foolkatGODOF_v3/ \
  --from_safetensors
```


## Inference

We provide a Gradio demo that integrates an SVG editor ([SVG-edit](https://edit.svgomg.net/)) and our processing interface.

### Linux Users

Use the following command:
```bash
python gradio_app.py
```

Then, open the `app.html` in the browser. Please use **Google Chrome**. 

### Windows Users

Please select a directory for placing the outputs first. Then, use the following command:
```bash
python gradio_app.py --data_base your/selected/directory
```

Afterwards, open the `app.html` in the browser. Remember to save the SVG as `untitled.svg` to that selected directory.


## Citation

If you use the code and models, please cite:

```bib
@article{mo2025doodleassist,
  title={DoodleAssist: Progressive Interactive Line Art Generation with Latent Distribution Alignment},
  author={Mo, Haoran and Shen, Yulin and Simo-Serra, Edgar and Wang, Zeyu},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements

This work is built based on [ControlNeXt](https://github.com/JIA-Lab-research/ControlNeXt) and the dataset [SketchMan](https://github.com/LCXCUC/SketchMan2020). We would like to thank their authors.

