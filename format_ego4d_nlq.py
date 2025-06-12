import json
import math
import csv
import os
import argparse
import bisect
from tqdm import tqdm
import random
random.seed(42)


def find_largest_le(sorted_list, target):
    """
    Returns the index of the largest value in sorted_list that is less than or equal to target.
    If no such element exists, returns -1.
    """
    # bisect_right returns an insertion point which is after any existing entries of target.
    index = bisect.bisect_right(sorted_list, target) - 1
    return index if index >= 0 else -1


def find_smallest_ge(sorted_list, target):
    """
    Returns the index of the smallest value in sorted_list that is greater than or equal to target.
    If no such element exists, returns -1.
    """
    index = bisect.bisect_left(sorted_list, target)
    return index if index < len(sorted_list) else -1

# python format_ego4d_nlq.py --data-path /datasets/ego4d/2025-04-02_1119_backup/v1/annotations/nlq_train.json --output-dir /home/shang/ego4d_nlq --num-frames 496 --extracted-fps 2 --max-segments 32 --augment-delay --num-epochs 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Path to the data directory")
    parser.add_argument("--output-path", type=str, help="Output path")
    parser.add_argument("--remote-path-to-frames", help="Path to frames on the hosts running the model")
    parser.add_argument("--num-frames", type=int, help="Number of frames to sample")
    parser.add_argument("--extracted-fps", type=int, default=2, help="Frames per second of the extracted frames")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs to generate")
    parser.add_argument("--augment-delay", action="store_true", help="Start the video from a later point in time as a form of data augmentation")
    parser.add_argument("--max-segments", type=int, default=1, help="Maximum number of question-answer pairs per datum")
    parser.add_argument("--max-length", type=float, default=None, help="The max length of the sampled video segment")
    parser.add_argument("--train", action="store_true", help="Flag to indicate that this is training data")
    args = parser.parse_args()

    with open(args.data_path) as f:
        nlq = json.load(f)
    data = []

    for epoch in tqdm(range(args.num_epochs)):
        train_epoch = []
        for video_index in tqdm(range(len(nlq['videos']))):
            for clip_index, clip in enumerate(nlq['videos'][video_index]['clips']):
                ann_count = 0
                current_datum = None
                video_start_sec = None
                video_end_sec = None
                frame_times_lst = None
                for lst_index, ann_list in enumerate(clip['annotations']):
                    ann_list['language_queries'] = sorted(ann_list['language_queries'], key=lambda x: x['video_start_sec'])
                    for a_index, ann in enumerate(ann_list['language_queries']):
                        if 'query' not in ann or ann['query'] is None: # if no query, skip
                            continue
                        started_new_datum = False
                        if current_datum is None:
                            video_start_sec = clip['video_start_sec']
                            video_end_sec = clip['video_end_sec']
                            if args.extracted_fps*(video_end_sec-video_start_sec) > args.num_frames and int(ann['video_start_sec']) > video_start_sec+1 and args.augment_delay and args.train: # Shift start for data augmentation
                                video_start_sec = video_start_sec + random.random() * min(video_end_sec-args.num_frames / args.extracted_fps - video_start_sec, (ann['video_start_sec']-1)-video_start_sec)
                                assert video_start_sec < ann['video_start_sec']
                            if args.max_length is not None and video_end_sec-video_start_sec > args.max_length:
                                video_start_sec = max(0, ann['video_start_sec']-args.max_length)+1 + random.random() * (args.max_length-2)
                                video_end_sec = video_start_sec + args.max_length
                                if video_end_sec > clip['video_end_sec']:
                                    video_end_sec = clip['video_end_sec']
                                    video_start_sec = video_end_sec - args.max_length
                            video_length_seconds = video_end_sec-video_start_sec # clip['video_start_sec']
                            video_num_frames = min(args.num_frames, int(args.extracted_fps * video_length_seconds)) # video_num_frames is the number of frames in the current clip
                            frame_times_lst = [i * video_length_seconds / video_num_frames for i in range(video_num_frames)] # frame_times list is the list of the frame times in seconds
                            # print("HERE3", video_start_sec)
                            # print("FRAME TIMES LST", frame_times_lst)
                            current_datum = {
                                'image_root': os.path.join(args.remote_path_to_frames, nlq['videos'][video_index]['video_uid'], nlq['videos'][video_index]['video_uid']),
                                'images': [int(args.extracted_fps*(t+video_start_sec))+1 for t in frame_times_lst],
                            }
                            started_new_datum = True
                            current_datum['question_id'] = f'ego4d_nlq_train_{video_index}_{clip_index}_{ann_count}'

                        start_sec = max(0, ann['video_start_sec'] - video_start_sec)
                        end_sec = max(ann['video_end_sec'] - video_start_sec, start_sec+1e-8)
                        assert end_sec > start_sec, str(start_sec)+', '+str(end_sec)
                        start = find_smallest_ge(frame_times_lst, start_sec)
                        end = find_largest_le(frame_times_lst, end_sec)
                        if start_sec > frame_times_lst[-1]:
                            assert start == -1
                            start = len(frame_times_lst)
                        if start > 0:
                            start -= 1
                        if end < len(frame_times_lst) - 1:
                            end += 1
                        assert -1 < start <= end
                        if end > args.num_frames or end == args.num_frames:
                            if started_new_datum:
                                current_datum = None
                            continue
                        assert start <= end
                        start = f"frame {start}"
                        end = f"frame {end}"
                        question = 'Find the video segment showing '+ann['query']
                        answer = f'Segment starts at {start} and ends at {end}.'

                        num_prev_segments = len([k for k in current_datum if 'answer' in k])
                        if args.max_segments == 1 or not args.train:
                            current_datum['question'] = question
                            current_datum['answer'] = answer
                        else:
                            current_datum['question'+str(num_prev_segments)] = question
                            current_datum['answer'+str(num_prev_segments)] = answer

                        if args.train and num_prev_segments+1 == args.max_segments:
                            train_epoch.append(current_datum)
                            current_datum = None
                        elif not args.train:
                            train_epoch.append(current_datum)
                            current_datum = None
                        ann_count += 1

                    if current_datum is not None:
                        num_prev_segments = len([k for k in current_datum if 'answer' in k])
                        assert args.train
                        train_epoch.append(current_datum)
                        current_datum = None
        if args.train:
            random.shuffle(train_epoch)
        data += train_epoch

    with open(os.path.join(args.output_path), 'w') as fout:
        for datum in data:
            fout.write(json.dumps(datum)+'\n')
