# *_*coding:utf-8 *_*
import os


############ For LINUX ##############
DATA_DIR = {
	'MER2023': '/home/wc/workspace/MER2023-Baseline/dataset-process',
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features/openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features'),
}
PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
}

PATH_TO_PRETRAINED_MODELS = '/home/samba/public/mer2023train/features/tools/transformers/'
PATH_TO_OPENSMILE = '/home/samba/public/mer2023train/features/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/home/samba/public/mer2023train/features/tools/ffmpeg-4.4.1-i686-static/ffmpeg'

SAVED_ROOT = os.path.join('./saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ (openface) ##############
DATA_DIR_Win = {
	'MER2023': '/home/wc/workspace/MER2023-Baseline/dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
}

PATH_TO_FEATURES_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
}

PATH_TO_OPENFACE_Win = "/home/hyc/MER/OpenFace/build/bin" #openface的路径