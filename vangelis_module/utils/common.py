

_PROJECT_ROOT = ''

_EMBEDDING_DIM = 50
_WINDOW = 10

_TRAINING_DATASET = ''


def set_project_root(project_root):
    global _PROJECT_ROOT
    _PROJECT_ROOT = project_root

def project_root():
    return _PROJECT_ROOT

def set_window(window):
    global _WINDOW
    _WINDOW = window

def window():
    return _WINDOW

def set_training_dataset(training_dataset):
    global _TRAINING_DATASET
    _TRAINING_DATASET = training_dataset

def training_dataset():
    return _TRAINING_DATASET



def set_embedding_dim(embedding_dim):
    global _EMBEDDING_DIM
    _EMBEDDING_DIM = embedding_dim

def embedding_dim():
    return _EMBEDDING_DIM