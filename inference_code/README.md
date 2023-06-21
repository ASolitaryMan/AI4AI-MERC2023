## Usage

### Prerequisites
- Python 3.8.16
- CUDA 11.7
- pytorch == 2.0.1
- torchvision == 0.15.2
- transformers == 4.29.1
- pandas == 1.5.3



### 
~~~~shell
## 第一次提交的投票的方法
python test.py --feat_path testset.scp --model_path /home/lqf/workspace/MERC2023workspace/new_train/fine_tune_wav --modal audio > audio_inference

# 这里的audio_inference 其实就是需要提交的结果，只是需要撰写为csv，下面的代码会统一转写

# 模型推理
python test.py --feat_path testset_src --model_path /home/lqf/workspace/MERC2023workspace/new_train/fine_tune_text --modal test > text_inference

## 第二次提交的投票的方法
python -u multi_test.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-4 --gpu=0 --batch_size=32 --save_root='./trimodal1/saved_att_lr4_bs32' --model_path=/home/wc/workspace/MER2023-Baseline/trimodal/saved_att_lr4_bs32/ 2>&1 | tee tri_att_bs32_lr4.log

## 第三次提交的投票的方法
python process_csv2.py ./trimodal/saved_att_lr4_bs32-trimodal/prediction/test3-pred-1686646362.6278818.csv > trimodel_inference

python vote.py audio_inference trimodel_inference text_inference
# 脚本运行完后，会生成三次结果，需要注意的是 audio_inference trimodel_inference text_inference 三者的顺序一定不能错
~~~~



### Acknowledgement
Thanks to [openface](https://github.com/TadasBaltrusaitis/OpenFace), [wenet](https://wenet.org.cn/wenet/), [pytorch](https://github.com/pytorch/pytorch),[Hugging Face](https://huggingface.co/docs/transformers/index)
