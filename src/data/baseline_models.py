import librosa
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from abc import ABC, abstractmethod

from vggish_utils import vggish_input_bk
from vggish_utils.vggish_model_architecture import VGGish2s
from scipy import spatial
import torch
from torch.autograd import Variable


class QueryByVoiceModel(ABC):
    '''
    Abstract base class for a query-by-voice machine learning model
    '''
    def __init__(
        self,
        model_filepath,
        parametric_representation,
        uses_windowing,
        window_length,
        hop_length):
        '''
        QueryByVoiceModel constructor.
        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.
            parametric_representation: A boolen. True if the audio
                representations depend on the model weights.
            uses_windowing: A boolean. Indicates whether the model slices the
                representation into fixed-length time windows.
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        #self.logger = get_logger('Model')

        self.model = None
        self.model_filepath = model_filepath
        self.parametric_representation = parametric_representation
        self.uses_windowing = uses_windowing
        self.window_length = window_length
        self.hop_length = hop_length

        self._load_model()

    @abstractmethod
    def construct_representation(self, audio_list, sampling_rates, is_query):
        '''
        Constructs the audio representation used during inference. Audio
        files from the dataset are constructed only once and cached for
        later reuse.
        Arguments:
            audio_list: A python list of 1D numpy arrays. Each array represents
                one variable-length mono audio file.
            sampling_rate: A python list of ints. The corresponding sampling
                rate of each element of audio_list.
            is_query: A boolean. True only if audio is a user query.
        Returns:
            A python list of audio representations. The list order should be
                the same as in audio_list.
        '''
        pass

    @abstractmethod
    def measure_similarity(self, query, items):
        '''
        Runs model inference on the query.
        Arguments:
            query: A numpy array. An audio representation as defined by
                construct_representation. The user's vocal query.
            items: A numpy array. The audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.
        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        pass

    @abstractmethod
    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make measure_similarityions.
        '''
        pass

    def _window(self, audio, sampling_rate):
        '''
        Chops the audio into windows of self.window_length seconds.
        Arguments:
            audio: A 1D numpy array. The audio to window.
            sampling_rate: An int. The sampling rate of the audio.
        Returns:
            A 2D numpy array of shape (windows, window_samples)
        '''
        window_samples = int(self.window_length * sampling_rate)
        hop_samples = int(self.hop_length * sampling_rate)

        if audio.shape[0] < window_samples:
            window = librosa.util.fix_length(audio, window_samples)
            return np.expand_dims(window, axis=0)
        else:
            return librosa.util.frame(audio, window_samples, hop_samples).T




class SiameseStyle(QueryByVoiceModel):
    '''
    A siamese-style neural network for query-by-voice applications.
    citation: Y. Zhang, B. Pardo, and Z. Duan, "Siamese Style Convolutional
        Neural Networks for Sound Search by Vocal Imitation," in IEEE/ACM
        Transactions on Audio, Speech, and Language Processing, pp. 99-112,
        2018.
    '''

    def __init__(
        self,
        model_filepath,
        parametric_representation=False,
        uses_windowing=True,
        window_length=4.0,
        hop_length=2.0):
        '''
        SiameseStyle model constructor.
        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.
            parametric_representation: A boolen. True if the audio
                representations depend on the model weights.
            uses_windowing: A boolean. Indicates whether the model slices the
                representation
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        super().__init__(
            model_filepath,
            parametric_representation,
            uses_windowing,
            window_length,
            hop_length)

    def construct_representation(self, audio_list, sampling_rates, is_query):
        '''
        Constructs the audio representation used during inference. Audio
        files from the dataset are constructed only once and cached for
        later reuse.
        Arguments:
            audio_list: A python list of 1D numpy arrays. Each array represents
                one variable-length mono audio file.
            sampling_rate: A python list of ints. The corresponding sampling
                rate of each element of audio_list.
            is_query: A boolean. True only if audio is a user query.
        Returns:
            A python list of audio representations. The list order should be
                the same as in audio_list.
        '''

        # Siamese-style network requires different representation of query
        # and dataset audio
        if is_query:
            representation = self._construct_representation_query(
                audio_list[0], sampling_rates[0])
        else:
            representation = self._construct_representation_dataset(
                audio_list, sampling_rates)
        return representation

    def measure_similarity(self, query, items):
        '''
        Runs model inference on the query.
        Arguments:
            query: A numpy array. An audio representation as defined by
                construct_representation. The user's vocal query.
            items: A numpy array. The audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.
        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        if not self.model:
            raise RuntimeError('No model loaded during call to \
                               measure_similarity.')

        # run model inference
        with self.graph.as_default():
            #self.logger.debug('Running inference')
            return np.array(self.model.predict(
                [query, items], batch_size=len(query), verbose=1),
                dtype='float64')

    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make predictions.
        '''
        #self.logger.info('Loading model weights from {}'.format(self.model_filepath))
        self.model = load_model(self.model_filepath)
        self.graph = tf.get_default_graph()

    def _construct_representation_query(self, query, sampling_rate):
        #self.logger.debug('Constructing query representation')

        # resample query at 16k
        new_sampling_rate = 16000
        query = librosa.resample(query, sampling_rate, new_sampling_rate)
        sampling_rate = new_sampling_rate

        if self.uses_windowing:
            windows = self._window(query, sampling_rate)
        else:
            windows = [
                librosa.util.fix_length(
                    query, self.window_length * sampling_rate)]

        # construct the logmelspectrogram of the signal
        representation = []
        for window in windows:
            melspec = librosa.feature.melspectrogram(
                window, sr=sampling_rate, n_fft=133,
                hop_length=133, power=2, n_mels=39,
                fmin=0.0, fmax=5000)
            melspec = melspec[:, :482]
            logmelspec = librosa.power_to_db(melspec, ref=np.max)
            representation.append(logmelspec)

        # normalize to zero mean and unit variance
        representation = np.array(representation)
        representation = self._normalize(representation).astype('float32')

        return [representation]

    def _construct_representation_dataset(self, dataset, sampling_rates):
        new_sampling_rate = 44100
        representations = []
        for audio, sampling_rate in zip(dataset, sampling_rates):

            # resample audio at 44.1k
            audio = librosa.resample(audio, sampling_rate, new_sampling_rate)
            sampling_rate = new_sampling_rate

            if self.uses_windowing:
                windows = self._window(audio, sampling_rate)
            else:
                windows = [
                    librosa.util.fix_length(
                        audio, self.window_length * sampling_rate)]

            representation = []
            for window in windows:
                # construct the logmelspectrogram of the signal
                melspec = librosa.feature.melspectrogram(
                    window,
                    sr=sampling_rate,
                    n_fft=1024,
                    hop_length=1024,
                    power=2)
                melspec = melspec[:, 0:128]
                logmelspec = librosa.power_to_db(melspec, ref=np.max)
                representation.append(logmelspec)

            # normalize to zero mean and unit variance
            representation = np.array(representation)
            representation = self._normalize(representation).astype('float32')
            representation = np.expand_dims(representation, axis=1)
            representations.append(representation)

        return representations

    def _normalize(self, x):
        # normalize to zero mean and unit variance
        mean = x.mean(keepdims=True)
        std = x.std(keepdims=True)
        return (x - mean) / std





class VGGishEmbedding(QueryByVoiceModel):
    '''
    A VGGish model to extract feature embeddings for query-by-voice applications.
    citation: S.Hershey,S.Chaudhuri,D.P.Ellis,J.F.Gemmeke,A.Jansen, R. C. Moore,
    M. Plakal, D. Platt, R. A. Saurous, B. Seybold, et al.,
    “Cnn architectures for large-scale audio classification,”
    in Acoustics, Speech and Signal Processing (ICASSP),
    2017 IEEE International Conference on. IEEE, 2017, pp. 131–135.
    '''

    def __init__(
        self,
        model_filepath,
        parametric_representation=False,
        uses_windowing=False,
        window_length=None,
        hop_length=None):
        '''
        SiameseStyle model constructor.
        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.
            parametric_representation: A boolen. True if the audio
                representations depend on the model weights.
            uses_windowing: A boolean. Indicates whether the model slices the
                representation
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        super().__init__(
            model_filepath,
            parametric_representation,
            uses_windowing,
            window_length,
            hop_length)

    def construct_representation(self, audio_list, sampling_rates, is_query):
        '''
        Constructs the audio representation used during inference. Audio
        files from the dataset are constructed only once and cached for
        later reuse.
        Arguments:
            audio_list: A python list of 1D numpy arrays. Each array represents
                one variable-length mono audio file.
            sampling_rate: A python list of ints. The corresponding sampling
                rate of each element of audio_list.
            is_query: A boolean. True only if audio is a user query.
        Returns:
            A python list of audio representations. The list order should be
                the same as in audio_list.
        '''
        pairs = zip(audio_list, sampling_rates)
        return [self._construct_representation(a, s) for (a, s) in pairs]

    def measure_similarity(self, query, items):
        '''
        Runs model inference on the query.
        Arguments:
            query: A numpy array. An audio representation as defined by
                construct_representation. The user's vocal query.
            items: A numpy array. The audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.
        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        if not self.model:
            raise RuntimeError('No model loaded during call to \
                               measure_similarity.')

        # run model inference
        #self.logger.debug('Running inference')
        simlarities = []
        for q, i in zip(query, items):
            simlarities.append(1 - spatial.distance.cosine(q, i))

        return np.array(simlarities)

    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make predictions.
        '''
        #self.logger.info('Loading model weights from {}'.format(self.model_filepath))
        self.model = VGGish2s()
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()

    def _construct_representation(self, audio, sampling_rate):
        # resample query at 16k
        new_sampling_rate = 16000
        audio = librosa.resample(audio, sampling_rate, new_sampling_rate)
        sampling_rate = new_sampling_rate

        # zero-padding
        target_length = int(np.ceil(audio.shape[0]/sampling_rate))
        if target_length % 2 != 0:
            target_length += 1
        pad = np.zeros((target_length*sampling_rate-audio.shape[0]))
        audio = np.append(audio, pad)

        melspec = vggish_input_bk.waveform_to_examples(audio, sampling_rate)
        melspec = melspec.astype('float32')
        representation = self.model(Variable(torch.from_numpy(melspec)))
        representation = representation.detach().numpy()

        return representation