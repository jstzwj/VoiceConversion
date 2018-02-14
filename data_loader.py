import params
import glob
import random
import librosa

def _get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(params.Default.sr, params.Default.n_fft, params.Default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs
    db = librosa.amplitude_to_db(mel)
    mfccs = np.dot(librosa.filters.dct(params.Default.n_mfcc, db.shape[0]), db)

    # Log
    mag = np.log(mag + sys.float_info.epsilon)
    mel = np.log(mel + sys.float_info.epsilon)

    # Normalization
    # self.y_log_spec = (y_log_spec - hp.mean_log_spec) / hp.std_log_spec
    # self.y_log_spec = (y_log_spec - hp.min_log_spec) / (hp.max_log_spec - hp.min_log_spec)

    return mfccs.T, mag.T, mel.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)

def get_mfccs_and_phones(wav_file, sr, trim=False, random_crop=True,
                         length=int(hp_default.duration / hp_default.frame_shift + 1)):
    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav, sr = librosa.load(wav_file, sr=sr)

    mfccs, _, _ = _get_mfcc_log_spec_and_log_mel_spec(wav, hp_default.preemphasis, hp_default.n_fft,
                                                      hp_default.win_length,
                                                      hp_default.hop_length)

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

def get_batch(mode, batch_size):
    '''Loads data.
    mode: A string. Either `train1` | `test1` | `train2` | `test2` | `convert`.
    '''

    if mode not in ('train1', 'test1', 'train2', 'test2', 'convert'):
        raise Exception("invalid mode={}".format(mode))

    with tf.device('/cpu:0'):
        # Load data
        wav_files = get_file_list(mode=mode)

        target_wavs = random.sample(wav_files, batch_size)

        if mode in ('train1', 'test1'):
            mfcc, ppg = map(_get_zero_padded, zip(*map(lambda w: get_mfccs_and_phones(w, hp_default.sr), target_wavs)))
            return mfcc, ppg
        else:
            mfcc, spec, mel = map(_get_zero_padded, zip(*map(
                lambda wav_file: get_mfccs_and_spectrogram(wav_file, duration=hp_default.duration), target_wavs)))
            return mfcc, spec, mel

def get_file_list(mode):
    '''Loads the list of sound files.
    mode: A string. One of the phases below:
      `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
      `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
      `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
      `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
      `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
    '''
    if mode == "train1":
        wav_files = glob.glob(params.Train1.data_path)
    elif mode == "test1":
        wav_files = glob.glob(params.Test1.data_path)
    elif mode == "train2":
        testset_size = params.Test2.batch_size * 4
        wav_files = glob.glob(params.Train2.data_path)[testset_size:]
    elif mode == "test2":
        testset_size =params.Test2.batch_size * 4
        wav_files = glob.glob(params.Train2.data_path)[:testset_size]
    elif mode == "convert":
        wav_files = glob.glob(params.Convert.data_path)
    return wav_files