import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = (
    4  # CCX-safe zone on this dual-CCX remote box (see project_sili_arch_sandbox_ccx_topology_thread_wall memory)
)
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
    "0.9",
    "0",
    "-1",
    "95.69",
    "-1",
    "0.9",
    "0",
    "2000",
    "0.5",
]
m.main()
