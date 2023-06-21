import pandas as pd


def load_csv(csv_path, save_path):
    df = pd.read_csv(csv_path)
    spk = df["utterance_id"].to_list()
    emo = df["emotion"].to_list()
    pred_list = []

    # read csv and sort
    for i in range(len(spk)):
        yp = spk[i].split("_")[1:]
        if len(yp[1]) == 1:
            yp[1] = str(0) + yp[1] 
        if len(yp[-1]) == 1:
            yp[-1] = str(0) + yp[-1] 
        spk_tmp = "_".join(yp)
        emotion = emo[i]
        pred_list.append(",".join([spk_tmp, emotion]))
    pred_list = sorted(pred_list)

    # write
    with open(save_path, 'w') as file:
        file.write("utterance_id,emotion" + '\n')
        for line in pred_list:
            tp = line.split(",")
            spk_id = tp[0].split("_")
            spk_id[1] = str(int(spk_id[1]))
            spk_id[-1] = str(int(spk_id[-1]))
            line = ",".join(["_".join(spk_id), tp[1].capitalize()])
            file.write(line + '\n')

    print(f"Process completed, the CSV file is saved at{save_path}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("python process_csv.py csv_path submission_save_path")
        sys.exit()
    csv_path = sys.argv[1]
    save_path = sys.argv[2]
    # csv_path = "/home/wc/workspace/MER2023-Baseline/trimodal/saved_att_lr4_bs32-trimodal/prediction/test3-pred-1686646362.6278818.csv"
    # save_path = "/home/wc/workspace/MER2023-Baseline/submission/submission.csv"
    load_csv(csv_path, save_path)