# f'ffmpeg -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" crop.mp4'
import json
import os
import subprocess
import sys

def run(params):
    # ffmpeg -i .\videos\id0001.mp4 -ss 4.533333333333333 -t 51.06666666666667 
    # -filter:v "crop=646:646:383:69, scale=1024:1024" crop.mp4
    cmd, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    # print(cmd)
    # return
    cmd_args = cmd.split(' ')
    cmd_args[-1]=os.path.split(cmd_args[2])[1].split('.')[0]+'_'+cmd_args[4]+'_'+cmd_args[6]+'.mp4'
    cmd_args[-1]=os.path.join(args.output_path, cmd_args[-1])
    # print(cmd_args[-1])
    # return
    cmd_args[8:10] = [' '.join(cmd_args[8:10])]
    cmd_args[8]=cmd_args[8].replace('"','')
    cmd_args.insert(1,'-y')
    # print(cmd_args)
    # return
    subprocess.call(cmd_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def get_sec(time_str):
    """Get seconds from time."""
    h, m, s, ss = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ss)+13)/60

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python cut_video.py dataset_path label_json_path save_path")
        sys.exit()

    video_path = sys.argv[1]
    data_json = sys.argv[2]
    output_path = sys.argv[3]

    if os.path.exists(output_path):
        os.makedirs(output_path)

    for folder in data_json:
        video_names = data_json[folder].keys()
        # videos = os.listdir(os.path.join(video_path, folder))
        for video in video_names:
            input_path = os.path.join(video_path, folder, video+'.mp4')
            dialog = data_json[folder][video.split('.')[0]]['Dialog']
            for segment in dialog:
                start = get_sec(dialog[segment]['StartTime'])
                end = get_sec(dialog[segment]['EndTime'])

                spk = dialog[segment]['Speaker']
                if not os.path.exists(os.path.join(output_path, folder)):
                    os.makedirs(os.path.join(output_path, folder))
                save_path = os.path.join(output_path, spk + "_" + segment +'.mp4')

                cmd = f'ffmpeg -y -i {input_path} -ss {start} -t {end-start} {save_path}'
                # cmd = f'ffmpeg -i {input_path} -ss {start} -t {end} -c copy {save_path}'
                # subprocess.call(cmd.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.call(cmd.split(' '))