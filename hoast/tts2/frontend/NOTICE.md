# Native Melo frontend third-party notices

This directory contains adaptations of English/Chinese normalization, phone
alignment and splitting from **MeloTTS**, copyright (c) 2024 MyShell.ai,
https://github.com/myshell-ai/MeloTTS, under the MIT license below.
Only EN/ZH text algorithms are adapted; upstream model wrappers are not used.

Mandarin tone-sandhi rules and the neutral-tone lexicon originate in
**PaddlePaddle/PaddleSpeech**, copyright (c) 2021 PaddlePaddle Authors.
All Rights Reserved. They are licensed under Apache License 2.0, reproduced in
`LICENSE-APACHE-2.0.txt`. The adaptation replaces implicit imports and mutable
segment representations, adds typing/documentation, and propagates errors
instead of swallowing them. Their license remains Apache 2.0; the project's
MIT license does not replace third-party licensing.

The NumPy English GRU inference algorithm and phoneme alphabet are adapted
from **g2p_en** by Kyubyong Park (kbpark.linguist@gmail.com) and Jongseok Kim
(https://github.com/ozmig77), https://github.com/Kyubyong/g2p, under Apache
License 2.0 (`LICENSE-APACHE-2.0.txt`). Modified for explicit local asset loading,
no auto-downloads, cached OOV predictions, validation, typing and documentation.
The separately prepared GRU weights and homograph table must retain their
upstream license and attribution when redistributed.

English number-normalization rules in Melo reference
https://github.com/keithito/tacotron (MIT); copyright (c) 2017 Keith Ito.

## CMU pronunciation dictionary

The Carnegie Mellon Pronouncing Dictionary [cmudict.0.6] is Copyright 1998
by Carnegie Mellon University. Use of this dictionary, for any research or
commercial purpose, is completely unrestricted. If you make use of or
redistribute this material, we would appreciate acknowledgement of its origin.

Source: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
The full original dictionary header must be retained in prepared `cmudict.rep`.
NLTK CMU resources are also separately prepared assets and must retain their
version-specific Carnegie Mellon University license/notice when redistributed.
No full dictionary or pretrained model is embedded in the source fixtures.

## MeloTTS MIT license

Copyright (c) 2024 MyShell.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Tacotron MIT license

Copyright (c) 2017 Keith Ito

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
