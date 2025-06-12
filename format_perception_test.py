from collections import defaultdict
import math
import csv
import json
import os
import argparse
import random
random.seed(42)
from tqdm import tqdm
import codecs

def unescaped_str(arg_str):
    return codecs.decode(str(arg_str), 'unicode_escape')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--remote-path-to-frames")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--extracted-fps", type=float, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--include-time-description", action="store_true") # Also use llava video prompt
    parser.add_argument("--choice-delimiter", default="\n", choices=[' ', '\n'], metavar="{ ,\\n}", type=unescaped_str)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--answer-prefix", default=None)
    args = parser.parse_args()

    letters = ['A', 'B', 'C', 'D']
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'valid']:
        if args.val_only and split == 'train':
            continue
        with open(os.path.join(args.data_dir, f'mc_question_{split}.json')) as f:
            data = json.load(f)
        formatted_data = []
        questions = []
        for key in data:
            for q in data[key]['mc_question']:
                q['metadata'] = data[key]['metadata']
                questions.append(q)
        if args.shuffle:
            random.shuffle(questions)
        for question in tqdm(questions):
            # datum = data[key]
            # for question in datum['mc_question']:
            choices = [f'({letter}) {choice}' for letter, choice in zip(letters, question['options'])]
            if args.include_time_description:
                choices = [f'{letter}. {choice}' for letter, choice in zip(letters, question['options'])]
            new_datum = {
                'question_id': question['metadata']['video_id']+'_'+str(question['id']),
                'question': 'Answer the question by choosing the correct option.\n'+question['question']+args.choice_delimiter+args.choice_delimiter.join(choices),
                'answer': choices[question['answer_id']],
            }
            if split != 'train':
                new_datum['answers'] = [new_datum['answer'], new_datum['answer'][1], new_datum['answer'][1]+'.', new_datum['answer'][1]+'. '+new_datum['answer'][4:], new_datum['answer'][1]+'. '+new_datum['answer'][4:]+'.']
                if args.include_time_description:
                    new_datum['answers'] = [new_datum['answer'][0], new_datum['answer'][0]+'.', new_datum['answer'], new_datum['answer']+'.']
                del new_datum['answer']
            video_length_seconds = question['metadata']['num_frames']/question['metadata']['frame_rate']
            total_video_frames = int(args.extracted_fps*video_length_seconds)
            new_datums = []
            interval = total_video_frames / args.num_frames
            total_count = 0
            for i in range(args.num_frames):
                if interval*i < total_video_frames:
                    new_datum[f'image_{total_count}'] = os.path.join(args.remote_path_to_frames, question['metadata']['video_id'], 'frame_{:04d}.jpg'.format(1+int(interval*i)))
                    total_count += 1
            new_datums.append(new_datum)
            # formatted_data.append(new_datum)
            if args.include_time_description:
                for datum in new_datums:
                    frame_times_lst = [
                        int(datum[f'image_{j}'].split('/')[-1].split('_')[1].split('.')[0]) / args.extracted_fps
                        for j in range(args.num_frames)
                        if f'image_{j}' in datum
                    ]
                    frame_times = ",".join([f"{t:.2f}s" for t in frame_times_lst])
                    time_description = f"The video lasts for {video_length_seconds:.2f} seconds, and {len(frame_times_lst)} frames are sampled from it. These frames are located at {frame_times}. Please answer the following questions related to this video.\nPlease respond with only the letter of the correct answer.\n"
                    datum['question'] = time_description+'\n'.join(datum['question'].split('\n')[1:])
                    to_add = ""
                    datum['question'] += to_add
            formatted_data += new_datums
        with open(os.path.join(args.output_dir, split+'.jsonl'), 'w') as fout:
            for datum in formatted_data:
                fout.write(json.dumps(datum)+'\n')
