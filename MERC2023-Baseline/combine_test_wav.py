import soundfile as sf
import os
import re
import numpy as np

def read_text(text_path):
    f = open(text_path, "r")
    lines = f.readlines()
    f.close()
    return lines


def combine_src(text_path, save_path):
    all_lines = read_text(text_path)
    
    num_lines = len(all_lines)

    index = 0
    while index < num_lines:
        text_list = []
        wav_list = []
        top_idx = []
        label_list = []
        while True:
            tmp = all_lines[index].strip().split("\n")[0].split()
            wav_src = tmp[0].split("_")
            spk, topic_idx = wav_src[0], wav_src[-2]

            if (index + 1) < num_lines:
                next_tmp = all_lines[index+1].strip().split("\n")[0].split()
                next_wav_src = next_tmp[0].split("_")
                next_spk, next_topic_idx = next_wav_src[0], next_wav_src[-2]
                # if next_spk == spk and topic_idx == next_topic_idx and tmp[-1] == next_tmp[-1]:
                if next_spk == spk and topic_idx == next_topic_idx:
                    #读文件，文本
                    test_str = re.search(r"\W", tmp[-1])
                    if test_str != None:
                        text_list.append(tmp[-1])
                    else:
                        text_list.append(tmp[-1] + "，")
                    
                    wav_name = "_".join(wav_src[:-1])
                    y, sr = sf.read(tmp[1])
                    wav_list.append(y)
                    top_idx.append(wav_src[-1])
                    # if tmp[-1] not in label_list:
                    #     label_list.append(tmp[-1])
                    index += 1
                else:
                    test_str = re.search(r"\W", tmp[-1])
                    if test_str != None:
                        text_list.append(tmp[-1])
                    else:
                        text_list.append(tmp[-1] + "。")

                    wav_name = "_".join(wav_src[:-1])
                    top_idx.append(wav_src[-1])
                    # if tmp[-1] not in label_list:
                    #     label_list.append(tmp[-1])

                    y, sr = sf.read(tmp[1])
                    wav_list.append(y)

                    text = "".join(text_list)
                    # label = " ".join(label_list)
                    tp = "_".join(top_idx)
                    save_file = save_path + wav_name + "_" + tp + ".wav"
                    sf.write(save_file, np.concatenate(wav_list), samplerate=sr)
                    print(" ".join([wav_name + "_" + tp, save_file, text]))
                    index += 1 
                    break
            else:
                index += 1
                break
    
    tmp = all_lines[-2].strip().split("\n")[0].split()
    wav_src = tmp[0].split("_")
    spk, topic_idx = wav_src[0], wav_src[-2]

    next_tmp = all_lines[-1].strip().split("\n")[0].split()
    next_wav_src = next_tmp[0].split("_")
    next_spk, next_topic_idx = next_wav_src[0], next_wav_src[-2]

    # if next_spk == spk and topic_idx == next_topic_idx and tmp[-1] == next_tmp[-1]:
    if next_spk == spk and topic_idx == next_topic_idx:
        text_list.append(next_tmp[-1])
        wav_name = "_".join(next_wav_src[:-1])
        top_idx.append(next_wav_src[-1])
        # if next_tmp[-1] not in label_list:
        #     label_list.append(next_tmp[-1])
        
        y, sr = sf.read(next_tmp[1])
        wav_list.append(y)
        text = ",".join(text_list)
        # label = " ".join(label_list)
        tp = "_".join(top_idx)
        save_file = save_path + wav_name + "_" + tp + ".wav"
        sf.write(save_file, np.concatenate(wav_list), samplerate=sr)
        print(" ".join([wav_name + "_" + tp, save_file, text]))
    else:
        tmp = all_lines[-1].strip().split("\n")[0].split()
        y, sr = sf.read(tmp[1])
        save_file = save_path + tmp[0] + ".wav"
        sf.write(save_file, y, sr)
        print(" ".join([tmp[0], save_file, tmp[-1]]))


if __name__ =="__main__":
    import sys
    if len(sys.argv) != 3:
        print("python combine_test_wav.py src_path combine_src_save_path")
        sys.exit()

    text_path = sys.argv[1]
    save_path = sys.argv[2]
    # text_path = "/home/lqf/workspace/MERC2023workspace/testset_src"
    # save_path = "/home/lqf/workspace/MERC2023workspace/audio/testset"
    if save_path[-1] != "/":
        save_path += "/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    combine_src(text_path, save_path)

