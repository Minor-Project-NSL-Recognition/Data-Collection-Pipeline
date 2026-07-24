# Notebooks

Exploratory / experiment notebooks. Keep them **thin**: import from the `nslr`
package and the `scripts/` logic — never redefine normalization, the model, or
other core logic here (that belongs in `nslr/`, which is diffable and will be
ported to JavaScript for the browser).

Start a notebook with:

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # repo root, so `import nslr` works
from nslr.dataset import load_processed
from nslr.preprocess import normalize_clip, standardize_length
```

Good candidates (preprocessing scope): frame-count distribution EDA, detection-
quality (`any_hand_detect_rate`) inspection, before/after normalization sanity
checks, per-signer clip-count balance.

> This is the `mid-defense-preprocessing` branch — model/training code
> (`nslr.model`, `train_eval.py`) lives on `master`.
