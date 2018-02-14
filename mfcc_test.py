import params
import glob
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy
import numpy
import sys
import tensorflow as tf

def load_vocab():
    phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
            'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
            'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
            'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
            'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

def preemphasis(x, coeff=0.97):
  return scipy.signal.lfilter([1, -coeff], [1], x)


def inv_preemphasis(x, coeff=0.97):
  return scipy.signal.lfilter([1], [1, -coeff], x)

def _get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = numpy.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(params.Default.sr, params.Default.n_fft, params.Default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = numpy.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs
    db = librosa.amplitude_to_db(mel)
    mfccs = numpy.dot(librosa.filters.dct(params.Default.n_mfcc, db.shape[0]), db)

    # Log
    mag = numpy.log(mag + sys.float_info.epsilon)
    mel = numpy.log(mel + sys.float_info.epsilon)

    # Normalization
    # self.y_log_spec = (y_log_spec - hp.mean_log_spec) / hp.std_log_spec
    # self.y_log_spec = (y_log_spec - hp.min_log_spec) / (hp.max_log_spec - hp.min_log_spec)

    return mfccs.T, mag.T, mel.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)

def get_mfccs_and_phones(wav_file, sr, trim=False, random_crop=True,
                        length=int(params.Default.duration / params.Default.frame_stride + 1)):
    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav, sr = librosa.load(wav_file, sr=sr)

    mfccs, _, _ = _get_mfcc_log_spec_and_log_mel_spec(wav, params.Default.preemphasis, params.Default.n_fft,
                                                      params.Default.win_length,
                                                      params.Default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r', errors='ignore').read().splitlines():
        if(len(line.split()) < 3):
            continue
        try:
            start_point, _, phn = line.split()
            bnd = int(start_point) // hp_default.hop_length
            phns[bnd:] = phn2idx[phn]
            bnd_list.append(bnd)
        except:
            continue
            # print('read error:\r\n')
            # print(line.split())
            # print('\r\n')
        

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - length)), 1)[0]
        end = start + length
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, length, axis=0)
    phns = librosa.util.fix_length(phns, length, axis=0)

    return mfccs, phns

def get_batch_queue(mode, batch_size):
    '''Loads data and put them in mini batch queues.
    mode: A string. Either `train1` | `test1` | `train2` | `test2` | `convert`.
    '''

    if mode not in ('train1', 'test1', 'train2', 'test2', 'convert'):
        raise Exception("invalid mode={}".format(mode))

    with tf.device('/cpu:0'):
        # Load data
        wav_files = get_files(mode=mode)

        # calc total batch count
        num_batch = len(wav_files) // batch_size

        filename_queue = tf.train.string_input_producer(wav_files)

        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.
        record_defaults = [[1], [1], [1], [1], [1]]
        col1, col2, col3, col4, col5 = tf.decode_csv(
            value, record_defaults=record_defaults)
        features = tf.concat(0, [col1, col2, col3, col4])

        if mode in ('train1', 'test1'):
            # Get inputs and target
            mfcc, ppg = get_mfccs_and_phones_queue(inputs=wav_file,
                                                   dtypes=[tf.float32, tf.int32],
                                                   capacity=2048,
                                                   num_threads=32)

            # create batch queues
            mfcc, ppg = tf.train.batch([mfcc, ppg],
                                       shapes=[(None, params.Default.n_mfcc), (None,)],
                                       num_threads=32,
                                       batch_size=batch_size,
                                       capacity=batch_size * 32,
                                       dynamic_pad=True)
            return mfcc, ppg, num_batch

def get_files(mode):
    '''Loads the list of sound files.
    mode: A string. One of the phases below:
      `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
      `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
      `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
      `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
      `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
    '''
    if mode == "train1":
        wav_files = glob.glob(hp.Train1.data_path)
    elif mode == "test1":
        wav_files = glob.glob(hp.Test1.data_path)
    elif mode == "train2":
        testset_size = hp.Test2.batch_size * 4
        wav_files = glob.glob(hp.Train2.data_path)[testset_size:]
    elif mode == "test2":
        testset_size = hp.Test2.batch_size * 4
        wav_files = glob.glob(hp.Train2.data_path)[:testset_size]
    elif mode == "convert":
        wav_files = glob.glob(hp.Convert.data_path)
    return wav_files

if __name__ == '__main__':
    wav, sr = librosa.load('voice1.wav', sr=44100)
    
    mfcc,_,_ = _get_mfcc_log_spec_and_log_mel_spec(wav, 0.97, 512, int(sr*0.005), int(sr*0.01))
    print(mfcc.shape)
    plt.plot(mfcc)
    plt.show()
    # librosa.display.specshow(mfcc, y_axis='linear')