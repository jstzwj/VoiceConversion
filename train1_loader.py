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

def get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
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
    # mag = numpy.log(mag + sys.float_info.epsilon)
    # mel = numpy.log(mel + sys.float_info.epsilon)

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

    # print(len(wav))

    mfccs, _, _ = get_mfcc_log_spec_and_log_mel_spec(wav, params.Default.preemphasis, params.Default.n_fft,
                                                      params.Default.win_length,
                                                      params.Default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = numpy.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r', errors='ignore').read().splitlines():
        if(len(line.split()) < 3):
            continue
        start_point, end_point, phn = line.split()
        bnd = int(start_point) // params.Default.hop_length
        
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    if random_crop:
        start = numpy.random.choice(range(numpy.maximum(1, len(mfccs) - length)), 1)[0]
        end = start + length
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, length, axis=0)
    phns = librosa.util.fix_length(phns, length, axis=0)

    return mfccs, phns


'''
        mfcc_batch = numpy.zeros((batch_size,params.Default.hop_length,params.Default.n_mfcc), dtype=int)
        ppg_batch = numpy.zeros((batch_size,params.Default.hop_length), dtype=int)

        for f in target_wavs:
            mfcc, ppg = get_mfccs_and_phones(f, params.Default.sr)
            print(mfcc.shape)
            print(ppg.shape)
            mfcc = numpy.reshape(mfcc,(1,params.Default.hop_length,params.Default.n_mfcc))
            mfcc_batch = numpy.concatenate((mfcc_batch, mfcc), 0)
            ppg = numpy.reshape(ppg,(1,params.Default.hop_length))
            ppg_batch = numpy.concatenate((ppg_batch, ppg), 0)
'''

def get_batch(batch_size):
    '''Loads data.
    '''

    with tf.device('/cpu:0'):
        # Load data
        wav_files = glob.glob(params.Train1.data_path)

        if len(wav_files) < batch_size:
            raise Exception("Number of wav files is {}. It is less than batch size.".format(len(wav_files)))

        target_wavs = random.sample(wav_files, batch_size)

        mfcc_batch, ppg_batch = zip(*map(lambda f:get_mfccs_and_phones(f, params.Default.sr), target_wavs))

        #print(mfcc_batch[0].shape)
        #print(ppg_batch[0].shape)

        # mfcc_batch = tf.convert_to_tensor(mfcc_batch, dtype=tf.float32)
        # ppg_batch = tf.convert_to_tensor(ppg_batch, dtype=tf.int32)
        #mfcc_batch, ppg_batch = tf.train.batch([mfcc, ppg],
        #                                shapes=[(None, params.Default.n_mfcc), (None,)],
        #                                num_threads=32,
        #                                batch_size=batch_size,
        #                                capacity=batch_size * 32,
        #                                dynamic_pad=True)
        return mfcc_batch, ppg_batch