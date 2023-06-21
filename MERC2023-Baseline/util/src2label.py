import sys

def src2laber(train_src_path, train_label_path):
    # read train_src and process
    output_lines = []
    with open(train_src_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(' ')
            if len(parts) >= 2:
                utterance = parts[0]
                wav_path = parts[1]

                utterance = utterance.split('/')[0]  # utterance
                emotion = parts[-1].lower()                  # emotion

                output_line = f"{utterance},{emotion}"
                output_lines.append(output_line)

    # 将转换后的结果写入新文件
    with open(train_label_path, 'w') as file:
        file.write("utterance_id,emotion" + '\n')
        for line in output_lines:
            file.write(line + '\n')

    print(f"Conversion completed, the CSV file is saved at{train_label_path}.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python src2label.py train_src_path train_label_path")
        sys.exit()
    # train_src_path = '/home/wc/workspace/MER2023-Baseline/src/train_src'                          # train_src path
    # train_label_path = '/home/wc/workspace/MER2023-Baseline/dataset-release/train-label.csv'  # ./dataset-release/train-label.csv

    train_src_path = sys.argv[1]                      # train_src path
    train_label_path = sys.argv[2]  # ./dataset-release/train-label.csv

    src2laber(train_src_path, train_label_path)
