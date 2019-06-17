
"""Hyperparameters. Most parameters do not need to be changed. Main tuning parameters:
1. num_speakers_per_batch
2. triplet_margin
3. quadruplet_margin
4. loss_type
5. margin
6. metric_model
7. sampling_type
8. lr
9. logdir (use a new directory to start an experiment.)
10. num_epochs."""


class Hyperparams:

    # TEDLIUM processing parameters. DO NOT CHANGE.
    tedlium_rate = 8000  # Sampling rate. If not 16kHz will do resampling.

    # CallHome processing parameters. DO NOT CHANGE.
    callhome_rate = 8000  # Sampling rate. If not 8kHz will do resampling.
    callhome_overlap = 0  # Segment overlap in sec (no overlapping for train).

    # MFCC parameters. DO NOT CHANGE.
    SOURCERATE = 1250
    TARGETRATE = 100000
    WINDOWSIZE = 250000.0
    frame_len = WINDOWSIZE / SOURCERATE / 8000
    frame_overlap = (WINDOWSIZE - TARGETRATE) / SOURCERATE / 8000
    mfcc_dim = 60  # Resulting dimension from other parameters in MFCC extraction.

    # Temporal segmentation parameters.
    seg_len = 2  # segment length in sec.
    win_len = int((seg_len - frame_overlap) / (frame_len - frame_overlap))  # num frames per segment.
    # since n*frame_len - (n-1)*frame_overlap = seg_len

    # Batch sampling parameters.
    batch_size = 256  # batch size of 256 is constrained by GPU memory 8GB.
    num_speakers_per_batch = 64  # Number of speakers to sample in each batch.
    
    #Parameters used for DWS
    batch_k = batch_size // num_speakers_per_batch
    cutoff = 0.5
    nonzero_loss_cutoff = 1.4
    
    # Metric learning model parameters.
    metric_model = 'trip'  # 'trip' or 'quad'
    loss_type = 'triplet' # Can use 'triplet' or 'quadruplet'. Used in conjunction with metric_model
    margin = 'fixed' # Can use 'adaptive'
    sampling_type = 'dws' #Can use 'random' or 'semihard' or 'dws'
    
    triplet_margin = 0.8  # alpha or alpha1 parameter in triplet loss/quadruplet loss respectively.
    quadruplet_margin = 0.4 # alpha2 parameter in quadruplet loss respectively.

    # Attention parameters.
    embed_type = '1DCNN'  # Input embedding. Options are 'raw', 'FCN', '1DCNN'.
    hidden_units = 256
    linear_units = 100  # Number of hidden units for the output linear layer.
    num_blocks = 2  # number of attention blocks.
    num_heads = 8  # number of parallel attention heads.

    # Training parameters.
    lr = 0.001  # learning rate.
    dropout_rate = 0.5
    num_epochs = 30
    logdir = 'experiments_70'  # Directory to store model and results.
    
    tedlium_src_dir = 'Data/TEDLIUM_release2/train/sph'      #Entire source .sph directory 
    tedlium_trsub_src_dir = 'Data/TEDLIUM_release2_subset/train/sph'  #Train subset .sph directory 
    tedlium_dev_src_dir = 'Data/TEDLIUM_release2_subset/dev/sph'  #Development set .sph directory    

    tedlium_target_dir = 'Data/TEDLIUM_release2/train/wav'  #Entire source .wav (target) directory
    tedlium_trsub_target_dir = 'Data/TEDLIUM_release2_subset/train/wav' #Train subset .wav(target) directory
    tedlium_dev_target_dir = 'Data/TEDLIUM_release2_subset/dev/wav'  #Development set .wav(target) directory
    tedlium_devdata_dir = 'Data/TEDLIUM_release2_subset/dev'
    tedlium_stm_dir = 'Data/TEDLIUM_release2/train/stm'  #TEDLIUM stm path
    
    callhome_src_dir = 'Data/CallHome/original'  #Entire CALLHOME source directory 
    callhome_target_dir = 'Data/CallHome/wav'    #CALLHOME target (wav) directory
    callhome_trsub_src_dir = 'Data/CallHome_subset/original' #CALLHOME subset source directory
    callhome_trsub_target_dir = 'Data/CallHome_subset/wav'  #CALLHOME subset target (wav) directory
    
    tedlium_trainsubset_dir = 'Data/TEDLIUM_release2_subset/train'
    tedlium_train_dir = 'Data/TEDLIUM_release2/train'
    
    callhome_data_dir = 'Data/CallHome'
    callhome_emb_dir = 'Data/CallHome/embs'
    callhome_label_dir = 'Data/CallHome/labels'
    callhome_pred_dir = 'Data/CallHome/predictions'
    
