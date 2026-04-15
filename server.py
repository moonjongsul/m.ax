import argparse

from common.utils import load_config




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="gt_kitting")
    args = parser.parse_args()

    cfg = load_config(fname='server_config.yaml', project=args.project)
    
    

if __name__ == "__main__":
    main()
