import sys
def src2transcription(src_path, trans_path):
    # read src and process
    output_lines = []
    with open(src_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(' ')
            if len(parts) >= 2:
                utterance = parts[0]
                wav_path = parts[1]

                utterance = utterance.split('/')[0]   # utterance
                if len(parts) == 3:
                    sentence = parts[-1]                  # sentence
                else:
                    sentence = parts[-2]                  # sentence

                output_line = f"{utterance},{sentence}"
                output_lines.append(output_line)

    # write transcription.csv
    with open(trans_path, 'w') as file:
        file.write("utterance_id,sentence" + '\n')
        for line in output_lines:
            file.write(line + '\n')

    print(f"Conversion completed, the CSV file is saved at{trans_path}.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python src2transcription.py train_src_path out_path")
        sys.exit()

    src_path = sys.argv[1]
    trans_path = sys.argv[2]
    # src_path = '/home/wc/workspace/MER2023-Baseline/src/testset_src'                           # src file path
    # trans_path = '/home/wc/workspace/MER2023-Baseline/dataset-process/transcription.csv'   # transcription path in config.py

    src2transcription(src_path, trans_path)
