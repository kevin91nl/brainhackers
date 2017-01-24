import os
import json

from .common import set_project_root
from .common import set_embedding_dim
from .common import set_training_dataset
from .common import set_window


_embedding_dim = 50
_window = 50
_project_dir = os.getcwd()
_config = json.load(open(os.path.join(_project_dir,'config.json')))
_embedding_dim = _config.get('embedding_dim',_embedding_dim)
_window = _config.get('window',_window)

set_embedding_dim(_embedding_dim)
set_training_dataset(os.path.join(_project_dir,'..\combined.pickle'))
set_window(_window)
set_project_root(_project_dir)

