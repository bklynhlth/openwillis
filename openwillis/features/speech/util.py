# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages

import os
import shutil
import piso

import pandas as pd
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()

def filter_rttm_line(line):
    line = line.decode('utf-8').strip()
    fields = line.split()
    
    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)
        
    file_id = fields[1]
    speaker_id = fields[7]
    
    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError:
        raise IOError('Turn duration not FLOAT. LINE: "%s"' % line)
        
    if dur <= 0:
        raise IOError('Turn duration <= 0 seconds. LINE: "%s"' % line)
        
    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError:
        raise IOError('Turn onset not FLOAT. LINE: "%s"' % line)
        
    if onset < 0:
        raise IOError('Turn onset < 0 seconds. LINE: "%s"' % line)
        
    return onset, dur, speaker_id, file_id

def load_rttm(rttmf):
    
    with open(rttmf, 'rb') as f:
        turns = []
        
        speaker_ids = set()
        file_ids = set()
        
        for line in f:
            if line.startswith(b'SPKR-INFO'):
                continue
                
            turn = filter_rttm_line(line)
            turns.append(turn)
    return turns

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def clean_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
        
def clean_prexisting(temp_dir, temp_rttm):
    #Clean prexisting dir
    clean_dir(temp_dir)
    clean_dir(temp_rttm)
    
def make_temp_dir(out_dir, temp_dir, temp_rttm):
    #Make dir
    make_dir(out_dir)
    make_dir(temp_dir)
    make_dir(temp_rttm)
    
def temp_process(out_dir, file_path, audio_path):
    temp_dir = os.path.join(out_dir, file_path + '_temp')
    temp_rttm = os.path.join(out_dir, file_path + '_rttm')
    
    clean_prexisting(temp_dir, temp_rttm)#clean dir
    make_temp_dir(out_dir, temp_dir, temp_rttm)#Make dir
    
    shutil.copy(audio_path, temp_dir)
    return temp_dir, temp_rttm

def overalp_index(df):
    df_combine = df.copy()
    
    if len(df)>1:
        com_interval = pd.IntervalIndex.from_arrays(df["start_time"], df["end_time"])
        
        df["isOverlap"] = piso.adjacency_matrix(com_interval).any(axis=1).astype(int).values
        df_n_overlap = df[df['isOverlap']==0]
        df_overlap = df[(df['isOverlap']==1) & (df['interval']>.5)]
        
        df_combine = pd.concat([df_n_overlap, df_overlap]).reset_index(drop=True)
    return df_combine
    
def read_rttm(temp_dir, file_path):
    rttm_df = pd.DataFrame()
    rttm_file = os.path.join(temp_dir, file_path + '.rttm')
    
    if os.path.exists(rttm_file): 
        rttm_info = load_rttm(rttm_file)

        rttm_df = pd.DataFrame(rttm_info, columns=['start_time', 'interval', 'speaker', 'filename'])
        rttm_df['end_time'] = rttm_df['start_time'] + rttm_df['interval']
        rttm_df = rttm_df.drop(columns=['filename'])

        rttm_df = overalp_index(rttm_df)
    return rttm_df

def audio_segment(audio_path, df_driad, out_dir):
    file_name, _ = os.path.splitext(os.path.basename(audio_path))
    
    aud_list = []
    speaker = df_driad['speaker'][0]
    
    sound = AudioSegment.from_wav(audio_path)
    for index, row in df_driad.iterrows():
        
        st_index = row['start_time']*1000
        end_index = row['end_time']*1000
        
        split_aud = sound[st_index:end_index+1]
        aud_list.append(split_aud)
        
    concat_audio = sum(aud_list)
    out_file = file_name + '_' +speaker +'.wav'
    concat_audio.export(os.path.join(out_dir, out_file), format="wav")
    
def slice_audio(df, audio_path, out_dir):
    speaker_list = list(df['speaker'].unique())[:2]
    
    for speaker in speaker_list:
        speaker_df = df[df['speaker']==speaker].reset_index(drop=True)
        audio_segment(audio_path, speaker_df, out_dir)
        
def prepare_diart_interval(start_time, end_time, speaker_list):
    df = pd.DataFrame(start_time, columns=['start_time'])
    
    df['end_time'] = end_time
    df['interval'] =df['end_time'] - df['start_time']
    df['speaker'] = speaker_list
    
    df = overalp_index(df)
    return df
        
def get_diart_interval(diarization):
    start_time = []
    end_time = []
    speaker_list = []
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        try:
            
            speaker_id = str(speaker).split('_')[1]
            speaker_id = int(speaker_id)
            start_time.append(turn.start)
            
            end_time.append(turn.end)
            speaker_list.append('speaker'+ str(speaker_id))
            
        except Exception as e:
            logger.info('Error in pyannote filtering')
            
    df = prepare_diart_interval(start_time, end_time, speaker_list)
    return df
