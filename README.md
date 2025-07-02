# S3QA

S3QA: Self-Supervised Speech Quality Assessment  
*A scalable method for automatically evaluating speech quality without human ratings*

This repository provides the model and inference code associated with the paper "Self‑Supervised Speech Quality Assessment (S3QA)" ([arXiv:2506.01655](https://arxiv.org/abs/2506.01655)). It enables users to apply the pretrained S3QA model to estimate speech quality from audio samples. The code includes utilities for preprocessing and running inference on custom audio, making it easy to integrate S3QA into evaluation workflows or downstream applications.

# Install
```
conda create -n s3qa_eval python=3.8.3
conda activate s3qa_eval
conda install pip
conda install anaconda::msgpack-python
pip install soundfile scipy librosa pandas pyloudnorm torch torchaudio lightning --timeout 100000
conda install -c conda-forge ffmpeg
```

# Example: Preprocessing
The resample script is an example of how to preprocess audio data. The S3QA model should only be run on 16k mono audio files (ideally wav to ensure compatibility).  

Example to resample an arbitrary audio path:   
```
python resample_data.py test_audio
```

# Example: Inference 
Example model inference for an arbitrary audio path:   
```
python run_model.py ckpts/small/BEST.ckpt test_audio_16k
```

# Example: Output
A CSV is produced that contains S3QA scores for each individual audio file.

Each row represents one utterance and includes the following columns:

- `fpath` | *Path to the audio file.*
- `s3qa_output_mean` | *Mean S3QA score across all analysis hops. Higher values typically indicate greater signal degradation.*
- `s3qa_output_median` | *Median S3QA score across hops, useful for reducing the influence of outliers.*
- `all_outputs_at_hop` | *A list of degradation scores at each analysis hop (e.g., frame/window-level scores). Useful for fine-grained temporal analysis.*

Example output from the run_model.py script:

```
fpath,s3qa_output_mean,s3qa_output_median,all_outputs_at_hop
test_audio_16k/SI860.wav,0.061343,0.061343,"[0.057289436, 0.06539719]"
```

# Tested Environments
16-inch 2021 MacBook Pro (M1 Pro 16GB Sequoia 15.3)

# Citations
If you use S3QA in your work, please cite:

```
@article{ogg2025self,
  title={Self-Supervised Speech Quality Assessment (S3QA): Leveraging Speech Foundation Models for a Scalable Speech Quality Metric},
  author={Ogg, Mattson and Bishop, Caitlyn and Yi, Han and Robinson, Sarah},
  journal={arXiv preprint arXiv:2506.01655},
  year={2025}
}
```

# Acknowledgements
This work was supported by internal research and development funding from the Johns Hopkins University Applied Physics Laboratory.

# License
See [LICENSE](LICENSE.md) for license information.

© 2025 The Johns Hopkins University Applied Physics Laboratory LLC
