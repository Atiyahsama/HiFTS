#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hifts.metrics import main
from hifts.traits import load_trait_map

if __name__ == "__main__":
    load_trait_map()
    main()
