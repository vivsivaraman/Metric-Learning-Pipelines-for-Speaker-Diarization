
"""For TEDLIUM, process from original sph audios to segments indexing. """

import csv
from glob import glob
import h5py
import numpy as np
from os import path, makedirs
from scipy.io import wavfile
from sox import Transformer
import sys
from hyperparams import Hyperparams as hp
from utils import parse_stm_file
from utils import extract_mfcc


def sph_to_wav(source_dir, target_dir):
    """Convert .sph files to .wav files."""

    assert path.exists(source_dir) is True

    if not path.exists(target_dir):
        makedirs(target_dir)

    for sph_file in glob(path.join(source_dir, "*.sph")):
        transformer = Transformer()
        if hp.tedlium_rate != 16000:
            transformer.set_output_format(encoding='signed-integer', channels=1, rate=hp.tedlium_rate)
        wav_filename = path.splitext(path.basename(sph_file))[0] + ".wav"
        wav_file = path.join(target_dir, wav_filename)
        transformer.build(sph_file, wav_file)


def compute_segment_mfcc(wav_dir, stm_dir, mfcc_dir):
    """Computer MFCC features for each recording and save to disk.
    Inputs:
        wav_dir: directory for wave files.
        stm_dir: directory for STM transcript files.
        mfcc_dir: directory for storing extracted MFCC features.
    """

    # Loop over stm files and split corresponding wav
    stm_files = glob(path.join(stm_dir, "*.stm"))
    for ii, stm_file in enumerate(stm_files):
        rec_name = path.splitext(path.basename(stm_file))[0]
        print('Processing ' + str(ii + 1) + ' / ' + str(len(stm_files)) + '     ' + rec_name)

        # Parse stm file
        stm_segments = parse_stm_file(stm_file)

        # Open wav corresponding to stm_file
        wav_filename = rec_name + ".wav"
        wav_file = path.join(wav_dir, wav_filename)
        rate, origAudio = wavfile.read(wav_file)

        # Loop over stm_segments and put together all audio signals from the speaker.
        speaker_data = []
        for stm_segment in stm_segments:
            # Create wav segment filename
            start_index = np.int64(stm_segment.start_time * rate)
            stop_index = np.int64(stm_segment.stop_time * rate)
            speaker_data.append(origAudio[start_index: stop_index + 1])
        speaker_data = np.concatenate(speaker_data)

        # Extract MFCC features.
        mfcc_features = extract_mfcc(speaker_data)
        print('MFCC feature dimension: ' + str(mfcc_features.shape))

        # Save to hdf5 binary files.
        if not path.exists(mfcc_dir):
            makedirs(mfcc_dir)
        with h5py.File(path.join(mfcc_dir, rec_name + '.hdf5'), 'w') as f:
            f.create_dataset('mfcc', data=mfcc_features)


def frames2segments(data_dir, save_dir):
    """Set up indexing so each row corresponds to segment [speaker label, recording, start index, end index]

    Inputs:
        data_dir: MFCC feature directory.
        save_dir: directory where the indexing text file will be saved as 'labels.csv'.
    """

    speaker_labels = {}  # speaker name : speaker ID
    speaker_id = 0
    all_info = []  # List of [label, recoding name, start index, end index]

    for txt_file in glob(path.join(data_dir, "*.hdf5")):

        rec_name = path.splitext(path.basename(txt_file))[0]  # Which recording the segment is from.

        # Get speaker label.
        speaker = rec_name.split('_')[0]
        if speaker in speaker_labels:
            label = speaker_labels[speaker]
        else:
            speaker_labels[speaker] = speaker_id
            label = speaker_id
            speaker_id += 1

        # Only need to get number of frames.
        with h5py.File(txt_file, 'r') as f:
            num_lines = f['mfcc'].len()

        num_frames_per_segment = hp.win_len
        # Estimate number of segments and skip the recording if too few segments.
        num_segments = num_lines // num_frames_per_segment
        if num_segments < 45:
            continue

        # Frame indices for the segment window
        start_index = 0
        end_index = num_frames_per_segment
        while end_index <= num_lines:
            # This completes all necessary information for the segment.
            all_info.append([label, rec_name, start_index, end_index])

            start_index = end_index + 1  # Non-overlapping segments, so skip the next frame.
            # Note this only applies to hp.frame_overlap == hp.frame_len / 2. TODO generalize this.
            end_index = start_index + num_frames_per_segment

    with open(path.join(save_dir, 'labels.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(all_info)



def main(args):
    """3 Steps in processing the TEDLIUM corpus:
        1. Convert .sph files to .wav files,
        2. Extract MFCC features for every wav file and save,
            2.1. For fixed-length short segments - to be used with attention model.
            2.2. Gather MFCC frames into segments (construct the segments indexing).
    """

    do_step_1 = True
    do_step_2_1 = True  
    do_step_2_2 = True  
        
    
    # Step 1.
    source = hp.tedlium_src_dir
    target = hp.tedlium_target_dir
    if do_step_1:
        sph_to_wav(source, target)

    # Step 2.1
    stm = path.join(path.dirname(target), 'stm')
    seg_feature_dir = path.join(path.dirname(target), 'mfcc_segments')  # Directory for MFCC features of all segments.
    if do_step_2_1:
        compute_segment_mfcc(target, stm, seg_feature_dir)

    # Step 2.2
    store_dir = path.dirname(seg_feature_dir)  # Save the label file to the parent directory.
    if do_step_2_2:
        frames2segments(seg_feature_dir, store_dir)

    
if __name__ == '__main__':
    main(sys.argv)
