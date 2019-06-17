
"""Process CallHome corpus."""

import csv
from glob import glob
import h5py
import numpy as np
from os import path, makedirs
from pyannote.core import Annotation, Segment, json
from scipy.io import wavfile
from sox import Transformer
import sys
from hyperparams import Hyperparams as hp
from utils import extract_mfcc


def compressed_wav_to_full(source_dir, target_dir):
    """Convert compressed wav files to full wav files."""

    assert path.exists(source_dir) is True

    if not path.exists(target_dir):
        makedirs(target_dir)

    for compressed_file in glob(path.join(source_dir, "*.wav")):
        transformer = Transformer()
        if hp.callhome_rate == 8000:
            transformer.set_output_format(encoding='signed-integer', channels=1)  # Also set single channel.
        else:  # Do resampling if specified.
            transformer.set_output_format(encoding='signed-integer', channels=1, rate=hp.callhome_rate)
        wav_filename = path.basename(compressed_file)
        wav_file = path.join(target_dir, wav_filename)
        transformer.build(compressed_file, wav_file)


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError as e:
        return False


def get_index(x):
    """From original CallHome index to sample index."""
    return x * 4 * 2


def parse_transcripts(f_path):
    lines = open(f_path, 'r').read().splitlines()
    uri = path.split(f_path)[1].split('.')[0]
    segments = []
    for i, line in enumerate(lines):
        if line.startswith('*'):
            speaker_id = line.split(':')[0][1:]
        splits = line.split(" ")
        if splits[-1].find('_') != -1:
            indexes = splits[-1].strip()
            start = indexes.split("_")[0].strip()[1:]
            end = indexes.split("_")[1].strip()[:-1]
            if represent_int(start) and represent_int(end):
                segments.append([uri, speaker_id, get_index(int(start)), get_index(int(end))])
    return segments


def get_transcripts(base_path):
    transcripts = []
    for f_path in glob(path.join(base_path, "*.cha")):
        transcripts.append(parse_transcripts(f_path))

    return transcripts


def compute_segments_mfcc(wav_dir, cha_dir, mfcc_dir, label_dir, pred_dir):
    """Computer MFCC features for each transcripted recording and save to disk.
    Inputs:
        wav_dir: directory for wave files.
        cha_dir: directory for cha transcript files.
        mfcc_dir: directory for storing segment MFCC features.
        label_dir: directory for saving ground-truth diarization, one file per conversation.
        pred_idr: directory for saving testing indices, one file per conversation.
    """

    # transcripts = []  # Stores ground-truth diarization results.
    # test_all_info = []  # Stores for every segment [recording name, start index, end index]

    # Loop over cha transcript files and get the segments.
    cha_files = glob(path.join(cha_dir, "*.cha"))
    for ii, cha_file in enumerate(cha_files):
        rec_name = path.splitext(path.basename(cha_file))[0]
        print('Processing ' + str(ii + 1) + ' / ' + str(len(cha_files)) + '     ' + rec_name)

        # Parse cha file to get ground-truth.
        cha_segments = parse_transcripts(cha_file)

        # To be consistent with literature, do the following additional processing:
        #   1. Use ground-truth segments as oracle SAD.
        #   2. Remove overlapping ground-truth segments.
        # As a result, MFCC will only be extracted within these regions.

        # Prepare ground-truth annotations.
        true_annotation = Annotation()
        for cha_segment in cha_segments:
            # Current segment.
            cur_seg = Segment(cha_segment[-2], cha_segment[-1])
            # Check for overlapping with all previous segments. If overlapping, remove both.
            overlap = False  # Indicator whether current overlaps with any of the previous.
            for prev_seg in true_annotation.itersegments():
                if prev_seg.intersects(cur_seg):
                    del true_annotation[prev_seg]
                    overlap = True
            if not overlap:  # Add to annotation only when no overlapping.
                true_annotation[cur_seg] = cha_segment[1]

        # Get wav audio corresponding to cha_file,
        # Use scipy instead of python wav since wav returns bytes array which is tricky to handle.
        wav_filename = rec_name + ".wav"
        wav_file = path.join(wav_dir, wav_filename)
        rate, origAudio = wavfile.read(wav_file)

        # Loop each SAD and extract MFCC and i-vectors.
        conv_mfcc = []  # Stores conversation level features.
        conv_pred_info = []  # Stores conversation level prediction segment indexing.
        for sad_region in true_annotation.itersegments():
            sad_start = sad_region[-2]
            sad_end = sad_region[-1]

            # Many ground-truth SAD regions are shorter than segment window. In this case take one segment.
            if sad_end < sad_start + hp.seg_len * rate:
                sad_end = sad_start + hp.seg_len * rate
            speaker_data = origAudio[sad_start: sad_end + 1]
            mfcc_features = extract_mfcc(speaker_data)

            # Within each SAD region, loop segment windows.
            start_frame_index = 0  # MFCC frame level.
            num_frames_per_segment = hp.win_len
            end_frame_index = num_frames_per_segment
            start_sample_index = 0  # Audio sample level.
            end_sample_index = hp.seg_len * rate
            while end_frame_index <= len(mfcc_features):
                conv_mfcc.append(mfcc_features[start_frame_index: end_frame_index, :])
                conv_pred_info.append([rec_name, sad_start + start_sample_index, sad_start + end_sample_index])
                # plus 'start' of the SAD region to get the global position

                # Move to next segment.
                move_frames = int(((hp.seg_len - hp.callhome_overlap) - hp.frame_overlap)
                                     / (hp.frame_len - hp.frame_overlap))  # Number of frames to pass.
                start_frame_index += (move_frames + 1)
                end_frame_index = start_frame_index + num_frames_per_segment

                # Move on sample level.
                move_samples = int((hp.seg_len - hp.callhome_overlap) * rate)
                start_sample_index += move_samples
                end_sample_index = start_sample_index + hp.seg_len * rate

        # Save to hdf5 binary files.
        if not path.exists(mfcc_dir):
            makedirs(mfcc_dir)
        conv_mfcc_ndarray = np.zeros((len(conv_mfcc), num_frames_per_segment, mfcc_features.shape[-1]))
        for i, feat in enumerate(conv_mfcc):
            conv_mfcc_ndarray[i] = feat
        print('Writing %d segments to disk..' % len(conv_mfcc))
        with h5py.File(path.join(mfcc_dir, rec_name + '.hdf5'), 'w') as f:
            f.create_dataset('mfcc', data=conv_mfcc_ndarray)

        # Save ground-truth annotations and prediction segment indexing.
        if not path.exists(label_dir):
            makedirs(label_dir)
        json.dump_to(true_annotation, path.join(label_dir, rec_name + '.json'))
        if not path.exists(pred_dir):
            makedirs(pred_dir)
        with open(path.join(pred_dir, rec_name + '.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(conv_pred_info)


def main(args):
    """2 Steps in processing the CALLHOME/CALLHOME_subset corpus:
        1. Convert compressed 2 channel wav files to uncompressed single channel wav,
        2. Extract MFCC features for every conversation, form the segments and save.
    """

    do_step_1 = True
    do_step_2 = True
    
    # Step 1.
    source = hp.callhome_src_dir
    target = hp.callhome_target_dir
    if do_step_1:
        compressed_wav_to_full(source, target)

    # Step 2.
    cha_dir = path.join(path.dirname(target), 'transcripts')
    feature_dir = path.join(path.dirname(target), 'mfcc')
    label_dir = path.join(path.dirname(target), 'labels')
    pred_dir = path.join(path.dirname(target), 'predictions')
    if do_step_2:
        compute_segments_mfcc(target, cha_dir, feature_dir, label_dir, pred_dir)
    

if __name__ == '__main__':
    main(sys.argv)
