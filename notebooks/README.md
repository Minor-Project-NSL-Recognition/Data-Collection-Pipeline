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
from nslr.model import build_bilstm
```

Good candidates: frame-count / detection-quality EDA, training curves and
confusion-matrix analysis, per-signer error inspection.
