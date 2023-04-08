import datetime


TRAIN_DATE_END = datetime.datetime(year=2019, month=10, day=1)
VAL_DATE_END = datetime.datetime(year=2020, month=1, day=1)

RAW_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]

LABEL_GROUPINGS = {
    'Atelectasis': ['Atelectasis'],
    'Cardiomegaly': ['Cardiomegaly'],
    'Consolidation': ['Consolidation', 'Pneumonia'],
    'Edema': ['Edema'],
    'Lung Lesion': ['Lung Lesion'],
    'No Finding': ['No Finding'],
    'Lung Opacity': ['Lung Opacity', 'Pneumonia', 'Consolidation', 'Lung Lesion', 'Atelectasis', 'Edmea'],
    'Pleural Other': ['Pleural Other', 'Pleural Effusion', 'Pneumothorax'],
    'Pleural Effusion': ['Pleural Effusion'],
    'Pneumonia': ['Pneumonia'],
    # 'Pneumothorax': ['Pneumothorax'],
    # 'Support Devices': ['Support Devices'],
    # 'Enlarged Cardiomediastinum': ['Enlarged Cardiomediastinum', 'Cardiomegaly'],
    # 'Fracture': ['Fracture'],
}
