import librosa
import numpy as np
import os
from abc import ABC, abstractmethod
from baseline_models import SiameseStyle, VGGishEmbedding

model_filepath = os.path.realpath('/Users/alejandrodelgadoluezas/Documents/GitHub/drum-sample-recommendation-by-vocal-imitation/model/baseline_weights/siamese_style.h5')
model_siamese = SiameseStyle(model_filepath)

model_filepath = os.path.realpath('/Users/alejandrodelgadoluezas/Documents/GitHub/drum-sample-recommendation-by-vocal-imitation/model/baseline_weights/vggish_pretrained_convs.pth')
model_vggish = VGGishEmbedding(model_filepath)



