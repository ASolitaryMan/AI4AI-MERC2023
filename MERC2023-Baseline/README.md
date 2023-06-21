## Usage

### Prerequisites
- Python 3.8.16
- CUDA 11.7
- pytorch == 2.0.1
- torchvision == 0.15.2
- transformers == 4.29.1
- pandas == 1.5.3
- 安装Linux版本的openface



### Build ./tools folder

```shell
## for visual feature extraction
https://drive.google.com/file/d/1DZVtpHWXuCmkEtwYJrTRZZBUGaKuA6N7/view?usp=share_link ->  tools/ferplus


## for audio extraction
https://www.johnvansickle.com/ffmpeg/old-releases ->  tools/ffmpeg-4.4.1-i686-static
## for acoustic acoustic features


## download wenet model and move to tools/wenet
visit "https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md" fill the request link and download
"https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz"

## huggingface for multimodal feature extracion
https://huggingface.co/TencentGameMate/chinese-hubert-large  -> tools/transformers/chinese-hubert-large
https://huggingface.co/hfl/chinese-macbert-large -> tools/transformers/chinese-macbert-large
```



### 
~~~~shell
# step1: dataset preprocess
python cut_video.py MERC_Challenge_CCAC2023_test_set MERC_Challenge_CCAC2023_test_set/test.json ./dataset-release/test3
python cut_video.py MERC_Challenge_CCAC2023_train_set MERC_Challenge_CCAC2023_train_set/train.json ./dataset-release/train
python cut_video.py MERC_Challenge_CCAC2023_test_set MERC_Challenge_CCAC2023_val_set/val.json ./dataset-release/val

python main-baseline.py normalize_dataset_format --data_root='./dataset-release' --save_root='./dataset-process'  2>&1 | tee preprocess.log

#acoustic
chmod -R 777 ./tools/ffmpeg-4.4.1-i686-static
chmod -R 777 ./tools/opensmile-2.3.0
python main-baseline.py split_audio_from_video_16k './dataset-process/video' './dataset-process/audio'  2>&1 | tee video2audio.log

python processing_json.py MERC_Challenge_CCAC2023_test_set/test.json ./data-process/audio/ test > ./src/test_src
python processing_json.py MERC_Challenge_CCAC2023_train_set/train.json ./data-process/audio/ train > ./src/train_src
python processing_json.py MERC_Challenge_CCAC2023_val_set/val.json ./data-process/audio/ val > ./src/val_src

cat ./src/train_src ./src/val_src  ./src/test_src > ./src/train_val_test_src

python util/src2label.py ./src/train_val_src ./data-release/train-label.csv
python util/src2transcription.py ./src/train_val_test_src ./data-process/transcription.csv

python combine_test_wav.py ./src/test_src ./audio/test > ./src/test.scp
python combine_wav.py ./src/train_src ./audio/train > ./src/train.scp
python combine_wav.py ./src/val_src ./audio/val > ./src/val.scp



# step2: multimodal feature extraction (see run_release.sh for more examples)
## you can choose feature_level in ['UTTERANCE', 'FRAME'] 
# visual
cd feature_extraction/visual
## set single-thread prevents crashes caused by multi-thread
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
## utilize the linux version openFace(change path and delete .exe in extract_openface.py)
python extract_openface.py --dataset=MER2023 --type=videoOne  2>&1 | tee openface.log 
## visual feature extraction(Resnet50-Ferplus)
python -u extract_ferplus_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0 2>&1 | tee video_feature.log

# audio
python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-large' --gpu=0  2>&1 | tee audio_feature.log
# text 
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'--gpu=0  2>&1 | tee text_feature.log

# step3: training unimodal and multimodal classifiers (see run_release.sh for more examples)
## multimodal
## 第二次提交的方法
python -u main-release.py --dataset='MER2023' --model_type='mlp' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-4 --gpu=0  --save_root='./trimodal/saved_mlp_lr4_bs32' 2>&1 | tee mlp_bs32_lr4.log

python -u main-release.py --dataset='MER2023' --model_type='att' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-4 --gpu=0  --save_root='./trimodal/saved_att_lr4_bs32' 2>&1 | tee att_bs32_lr4.log

# 运行完成后，在当前目录下会生成一个trimodal的文件夹，层层打开进入prediction文件夹，找到一个csv文件，假设文件路径为./trimodal/saved_att_lr4_bs32-trimodal/prediction/test3-pred-1686646362.6278818.csv
python util/process_csv.py ./trimodal/saved_att_lr4_bs32-trimodal/prediction/test3-pred-1686646362.6278818.csv ./submission/submission.csv

## 第一次提交的方法
## 四卡，开启fp16
# 模型训练
accelerate launch --mixed_precision="fp16" fine_tune_hubert.py --save_path ./save_model --train_src ./src/train.scp --valid_src ./src/val.scp --pre_train_model ./tools/transformers/chinese-hubert-large
# 模型推理
python test.py --feat_path ./src/test.scp --model_path /home/lqf/workspace/MERC2023workspace/new_train/fine_tune_wav --modal audio > audio_inference

# 这里的audio_inference 其实就是需要提交的结果，只是需要撰写为csv，下面的代码会统一转写

## 文本fine-tune的方法
accelerate launch --mixed_precision="fp16" fine_tune_macbert.py --save_path ./save_model --train_src ./src/train.scp --valid_src ./src/val.scp --pre_train_model ./tools/transformers/chinese-hubert-large
# 模型推理
python test.py --feat_path ./src/test_src --model_path /home/lqf/workspace/MERC2023workspace/new_train/fine_tune_text --modal test > text_inference

## 第三次提交的投票的方法
python util/process_csv2.py ./trimodal/saved_att_lr4_bs32-trimodal/prediction/test3-pred-1686646362.6278818.csv > trimodel_inference

python vote.py audio_inference trimodel_inference text_inference
# 脚本运行完后，会生成三次结果，需要注意的是 audio_inference trimodel_inference text_inference 三者的顺序一定不能错
~~~~



### Other Examples
For other datasets, please refer to **run_release.sh**



### Acknowledgement
Thanks to [openface](https://github.com/TadasBaltrusaitis/OpenFace), [wenet](https://wenet.org.cn/wenet/), [pytorch](https://github.com/pytorch/pytorch),[Hugging Face](https://huggingface.co/docs/transformers/index)
