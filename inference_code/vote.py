import os
import pandas as pd
from collections import Counter

#utterance_id, emotion
def read_text(text_path):
    f = open(text_path, 'r')
    lines = f.readlines()
    f.close()

    return lines

def voting(pred_test_2, pred_test3, pred_text):
    test2 = read_text(pred_test_2)
    test3 = read_text(pred_test3)
    text = read_text(pred_text)
    name = []
    test2_label = []
    test3_label = []
    vote_label = []

    for i in range(len(test2)):
        l2 = test2[i].strip().split("\n")[0].split()
        l3 = test3[i].strip().split("\n")[0].split()
        l4 = text[i].strip().split("\n")[0].split()

        assert(l2[0] == l3[0] == l4[0])
        name.append(l2[0])
        test2_label.append(l2[1])
        l3_lab = l3[1].capitalize()
        test3_label.append(l3_lab)
        tmp = [l2[1], l3_lab, l4[1]]
        result = Counter(tmp)
        max = 0
        lb = ""
        for key in result:
            if result[key] > max:
                max = result[key]
                lb = key
        if max == 1:
            vote_label.append(l2[1])
        else:
            vote_label.append(lb)
        max = 0

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()

    df1["utterance_id"] = name
    df1["emotion"] = test2_label

    df2["utterance_id"] = name
    df2["emotion"] = test3_label

    df3["utterance_id"] = name
    df3["emotion"] = vote_label

    df1.to_csv("第一次结果.csv", sep=",", encoding="utf-8", index=False)
    df2.to_csv("第二次结果.csv", sep=",", encoding="utf-8", index=False)
    df3.to_csv("第三次结果.csv", sep=",", encoding="utf-8", index=False)



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("python vote.py audio_inference multimodal_inference text_inference")
        sys.exit()
    p2 = sys.argv[1]
    p3 = sys.argv[2]
    pt = sys.argv[3]
    # p2 = "/home/lqf/workspace/MERC2023workspace/pred_test_2"
    # p3 = "/home/lqf/workspace/MERC2023workspace/pred_test_4"
    # pt = "/home/lqf/workspace/MERC2023workspace/pred_text"
    voting(p2,p3,pt)
