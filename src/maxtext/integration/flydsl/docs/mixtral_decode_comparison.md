# Decode comparison report: mixtral-8x7b

Validation that the FlyDSL MoE backend (`moe_backend=flydsl`) matches MaxText's
exact `ragged_dot` baseline on real weights, via the real KV-cache autoregressive
decode path. Run with the offline-preshuffled checkpoint and `.flydsl_env` sourced
(gfx950 / MI355X).

- seq=512, gen_steps=16, scan_layers=False, greedy, dtype=bfloat16, single device
- backends: ragged (baseline), fly (FlyDSL)
- weights: preshuffled mixtral-8x7b checkpoint (offline-converted from the stock checkpoint)
- Greedy autoregressive decode via the real KV-cache path; compares generated token sequences.

## ragged vs fly

| # | prompt | leading match | exact? | ragged text | fly text |
|---|--------|---------------|--------|------|------|
| 0 | 'The capital of France is' | 16/16 | Y | 'a city that is known for its beauty and its culture. It is also a' | 'a city that is known for its beauty and its culture. It is also a' |
| 1 | 'In a galaxy far, far away,' | 16/16 | Y | 'a long time ago, a young man named George Lucas was inspired by a Japanese' | 'a long time ago, a young man named George Lucas was inspired by a Japanese' |
| 2 | 'def fibonacci(n):' | 16/16 | Y | '\n    if n == 0:\n        return 0\n    elif' | '\n    if n == 0:\n        return 0\n    elif' |
| 3 | 'Photosynthesis is the process by which' | 16/16 | Y | 'plants convert light energy into chemical energy. This chemical energy is stored in the bonds' | 'plants convert light energy into chemical energy. This chemical energy is stored in the bonds' |
| 4 | 'The quick brown fox jumps over the' | 16/16 | Y | 'lazy dog.\n\nThe quick brown fox jumps over the lazy dog' | 'lazy dog.\n\nThe quick brown fox jumps over the lazy dog' |
| 5 | 'Once upon a time, there was a' | 16/16 | Y | 'little girl who loved to read. She read everything she could get her hands on' | 'little girl who loved to read. She read everything she could get her hands on' |
| 6 | 'The first president of the United States was' | 16/16 | Y | 'George Washington. He was born on February 22, 173' | 'George Washington. He was born on February 22, 173' |
| 7 | 'Water boils at a temperature of' | 16/16 | Y | '100 degrees Celsius.\n\n## What is the temperature' | '100 degrees Celsius.\n\n## What is the temperature' |
| 8 | 'The largest planet in our solar system is' | 16/16 | Y | 'Jupiter. It is the fifth planet from the sun. Jupiter is a' | 'Jupiter. It is the fifth planet from the sun. Jupiter is a' |
| 9 | 'import numpy as np' | 16/16 | Y | '\nimport matplotlib.pyplot as plt\nimport pandas as pd' | '\nimport matplotlib.pyplot as plt\nimport pandas as pd' |

**fly: 10/10 sequences exactly match ragged.**
