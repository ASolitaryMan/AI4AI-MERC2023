import json
import os
import sys

def load_json(json_path, root_wav):
    with open(json_path,'r',encoding = 'utf-8') as fp:
        data_json = json.load(fp)
        for key in data_json:
            file_src = data_json[key]
            for dialog_idx in file_src:
                dialog_src = file_src[dialog_idx]["Dialog"]
                for sent in dialog_src:
                    text = dialog_src[sent]["Text"]
                    text = text.replace(" ", "，")
                    spk = dialog_src[sent]["Speaker"]
                    # emotion = dialog_src[sent]["EmoAnnotation"]["final_main_emo"]
                    print(spk+"_"+sent, root_wav + spk+"_"+sent +".wav", text)

def load_json_train(json_path, root_wav):
    with open(json_path,'r',encoding = 'utf-8') as fp:
        data_json = json.load(fp)
        for key in data_json:
            file_src = data_json[key]
            for dialog_idx in file_src:
                dialog_src = file_src[dialog_idx]["Dialog"]
                for sent in dialog_src:
                    text = dialog_src[sent]["Text"]
                    text = text.replace(" ", "，")
                    spk = dialog_src[sent]["Speaker"]
                    emotion = dialog_src[sent]["EmoAnnotation"]["final_main_emo"]
                    print(spk+"_"+sent, root_wav + spk+"_"+sent +".wav", text, emotion)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python processing_json.py label_json_path data_path mode")
        sys.exit()

    test_path = sys.argv[1]
    root_wav = sys.argv[2]
    mode = sys.argv[3]

    if mode == "test":
        load_json(test_path, root_wav)
    else:
        load_json_train(test_path, root_wav)