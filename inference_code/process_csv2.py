import pandas as pd
import os

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    spk = df["utterance_id"].to_list()
    emo = df["emotion"].to_list()

    pred_list = []

    for i in range(len(spk)):
        yp = spk[i].split("_")[1:]
        if len(yp[1]) == 1:
            yp[1] = str(0) + yp[1] 
        if len(yp[-1]) == 1:
            yp[-1] = str(0) + yp[-1] 
        spk_tmp = "_".join(yp)
        emotion = emo[i]
        pred_list.append(" ".join([spk_tmp, emotion]))
    
    pred_list = sorted(pred_list)

    for item in pred_list:
        tp = item.split()
        spk_id = tp[0].split("_")
        spk_id[1] = str(int(spk_id[1]))
        spk_id[-1] = str(int(spk_id[-1]))
        print(" ".join(["_".join(spk_id), tp[1]]))



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("python vote.py csv_path")
        sys.exit()
    
    csv_path = sys.argv[1]
    load_csv(csv_path)