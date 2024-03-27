import os
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import python_speech_features as psf
from preprocessing import get_fbanks

np.random.seed(42)


def read_metadata(root,audios):
    speakers = pd.read_csv(root+'/SPEAKERS.TXT', sep='|', skip_blank_lines=True)
    speakers_filtered = speakers[(speakers['SUBSET'] == audios)]
    speakers_filtered = speakers_filtered.copy()
    speakers_filtered['LABEL'] = speakers_filtered['ID'].astype('category').cat.codes
    speakers_filtered = speakers_filtered.reset_index(drop=True)
    return speakers_filtered



def assert_out_dir_exists(index,OUTPUT_PATH):
    dir_ = OUTPUT_PATH + '/' + str(index)

    if not os.path.exists(dir_):
        os.makedirs(dir_)
        print('crated dir {}'.format(dir_))
    else:
        print('dir {} already exists'.format(dir_))

    return dir_


def main(root= 'datasets/LibriSpeech',audios='audio',type_audio='flac',OUTPUT_PATH='flask-test'):
    speakers = read_metadata(root,audios)

    print('read metadata from file, number of rows in in are: {}'.format(speakers.shape))
    print('numer of unique labels in the dataset is: {}'.format(speakers['LABEL'].unique().shape))
    print('max label in the dataset is: {}'.format(speakers['LABEL'].max()))
    print('number of unique index: {}, max index: {}'.format(speakers.index.shape, max(speakers.index)))

    for index, row in speakers.iterrows():
        subset = row['SUBSET']
        id_ = row['ID']
        dir_ = root + '/' + subset + '/' + str(id_) + '/'

        print('working for id: {}, index: {}, at path: {}'.format(id_, index, dir_))

        files_iter = Path(dir_).glob('**/*.'+type_audio)
        files_ = [str(f) for f in files_iter]

        index_target_dir = assert_out_dir_exists(index,OUTPUT_PATH)

        sample_counter = 0

        for f in files_:
            fbanks = get_fbanks(f)
            num_frames = fbanks.shape[0]

            # sample sets of 64 frames each
            file_sample_counter = 0
            start = 0
            while start < num_frames + 64:
                slice_ = fbanks[start:start + 64]
                if slice_ is not None and slice_.shape[0] == 64:
                    assert slice_.shape[0] == 64
                    assert slice_.shape[1] == 64
                    assert slice_.shape[2] == 1
                    np.save(index_target_dir + '/' + str(sample_counter) + '.npy', slice_)

                    file_sample_counter += 1
                    sample_counter += 1

                start = start + 64

            print('done for index: {}, Samples from this file: {}'.format(index, file_sample_counter))

        print('done for id: {}, index: {}, total number of samples for this id: {}'.format(id_, index, sample_counter))
        print('')

    print('All done, writing to file')

    print('All done, YAY!, look at the files')


if __name__ == '__main__':
    main(audios='audio-test',OUTPUT_PATH='fbank-test')
