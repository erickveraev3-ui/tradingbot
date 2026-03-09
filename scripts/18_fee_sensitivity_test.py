import subprocess

fees = [0.0004, 0.0006, 0.0008, 0.0010]

for fee in fees:
    print("\n===================================")
    print(f"TESTING FEE = {fee}")
    print("===================================\n")

    cmd = [
        "python",
        "scripts/08_backtest_meta_model_v6.py",
        "--fee", str(fee)
    ]

    subprocess.run(cmd)
