import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

sys.argv = [
    "train_mqar_curriculum.py",
    "fp32",
    "100000",
    "1000",
    "0.015",
    "16",
    "10",
    "1",
    "1",
    "50",
    "0",
    "0",
    "36",
    "-1",
    "-1",
    "-1",
    "0",
    "-1",
    "100000000",
    "-1",
    "0",
    "-1",
    "-1",
    "-1",
    "-1",
    "0",
    "-1",
    "-1",
]
m.main()
