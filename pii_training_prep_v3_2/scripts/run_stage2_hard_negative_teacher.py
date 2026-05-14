from pii_prep.stage2_hard_negative_teacher import main
import sys


if __name__ == "__main__":
    raise SystemExit(main(["--run-teacher", *sys.argv[1:]]))
