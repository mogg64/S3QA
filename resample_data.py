"""
Copyright © 2025 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import sys
import torchaudio

data_path = sys.argv[1]
if not data_path.endswith(os.path.sep):
    data_path += os.path.sep

print('starting: ', data_path)
print('collecting audio...')

audio_files = []
for xx in Path(data_path).glob("**/*.wav"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.WAV"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.mp3"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.flac"):
    audio_files.append(xx)

print('collected audio files: ', str(len(audio_files)))
print('resampling audio files!')

for idx, f in enumerate(audio_files):

    in_path = str(audio_files[idx])
    in_ext = os.path.splitext(in_path)[1]
    out_path = in_path.replace(os.path.dirname(data_path), os.path.dirname(data_path) + '_16k')
    out_path = out_path.replace(in_ext, '.wav')

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    # processing for mp3s:
    # mp3s can be hard for librosa to load, but easier for torchaudio to load
    # temporarily load it as a .mp3, then write to a .wav and continue the rest of the pipeline
    # delete the temp file at the end
    is_mp3 = False
    if in_ext == '.mp3':
        audio_tmp, sr_tmp = torchaudio.load(in_path)
        torchaudio.save(in_path.replace('.mp3', '.wav'), audio_tmp, sr_tmp)
        in_path = in_path.replace('.mp3', '.wav')
        is_mp3 = True

    y, sr = librosa.load(in_path, sr=16000, mono=True)
    if len(y) > sr * 10:
        y = y[:(sr * 10)]

    sf.write(out_path, y, sr)

    # if we made a temp file to deal with an mp3, delete the temp file
    if is_mp3:
        os.remove(in_path)

    print('done! ', in_path)
