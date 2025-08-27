import os
import pandas as pd
import numpy as np
from phe import paillier
import argparse
import pickle
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./dataset/xw/elec/test")
    parser.add_argument("--output_dir", type=str, default="./dataset/dec")
    parser.add_argument("--key_size", type=int, default=2048)
    parser.add_argument("--key_path", type=str, default="./keys")
    parser.add_argument("--task", type=str, default="enc", choices=["key_gen", "enc", "dec"])
    args = parser.parse_args()

    if args.task == "key_gen":
        os.makedirs(args.key_path, exist_ok=True)
        print("Generating keys...")
        pubkey, privkey = paillier.generate_paillier_keypair(n_length=args.key_size)
        with open(os.path.join(args.key_path, "pubkey.pkl"), "wb") as f:
            pickle.dump(pubkey, f)
        with open(os.path.join(args.key_path, "privkey.pkl"), "wb") as f:
            pickle.dump(privkey, f)
        print("Keys generated and saved to", args.key_path)

    if args.task == "enc":
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.key_path, "pubkey.pkl"), "rb") as f:
            pubkey = pickle.load(f)
        print("Encrypting data...")
        for file in os.listdir(args.input_dir):
            if file.endswith(".csv"):
                print("Encrypting", file)
                df = pd.read_csv(os.path.join(args.input_dir, file))
                if "time" in df.columns:
                    columns = df.columns[1:].tolist()
                else:
                    columns = df.columns.tolist()
                for col in columns:
                    df[col] = df[col].apply(lambda x: (pubkey.encrypt(x).ciphertext(), pubkey.encrypt(x).exponent))
                df.to_csv(os.path.join(args.output_dir, file), index=False)

    elif args.task == "dec":
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.key_path, "pubkey.pkl"), "rb") as f:
            pubkey = pickle.load(f)
        with open(os.path.join(args.key_path, "privkey.pkl"), "rb") as f:
            privkey = pickle.load(f)
        print("Decrypting data...")
        for file in os.listdir(args.input_dir):
            if file.endswith(".csv"):
                print("Decrypting", file)
                df_columns = pd.read_csv(os.path.join(args.input_dir, file), nrows=1).columns.tolist()
                if 'time' in df_columns:
                    df_columns.remove('time')
                df = pd.read_csv(os.path.join(args.input_dir, file), converters={
                                    col: ast.literal_eval
                                    for col in df_columns
                                })
                if "time" in df.columns:
                    columns = df.columns[1:].tolist()
                else:
                    columns = df.columns.tolist()
                for col in columns:
                    df[col] = df[col].apply(lambda x: privkey.decrypt(paillier.EncryptedNumber(pubkey, x[0], x[1])))
                df.to_csv(os.path.join(args.output_dir, file), index=False)




