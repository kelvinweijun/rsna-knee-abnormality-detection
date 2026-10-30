import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "trainer_moaks.py")


def run(name, args):
    cmd = [sys.executable, "-u", TRAINER] + args
    print("[stage] %s: %s" % (name, " ".join(cmd)), flush=True)
    t = time.time()
    rc = subprocess.run(cmd).returncode
    print("[stage] %s rc=%d %.1f min" % (name, rc, (time.time() - t) / 60), flush=True)
    return rc


def main():
    ck = os.path.join(HERE, "raptor_ft_moaks384.pt")
    rc = run("pretrain", ["--tag", "moaks384", "--epochs", "14"])
    if rc != 0 or not os.path.exists(ck):
        print("[stage] pretrain failed or checkpoint missing - stopping", flush=True)
        return 1
    return run("finetune", ["--tag", "moaksft384", "--epochs", "8", "--init", ck, "--oai_dir", ""])


if __name__ == "__main__":
    sys.exit(main())
