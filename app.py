from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
#nitin
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request,session,redirect,flash,url_for
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from datetime import datetime
import json
import os
import pandas as pd
from werkzeug.utils import secure_filename

# import count_vect

from flask import Flask, jsonify, request
import numpy as np

import pandas as pd
import numpy as np





