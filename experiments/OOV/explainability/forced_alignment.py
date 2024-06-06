import os
from dataclasses import dataclass
import torch
import torchaudio
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()

dictionary = {c: i for i, c in enumerate(labels)}


def extract_features(segmented_audio, segmented_labels, filename, split):
    combined_features = []
    combined_labels = []
    for i in range(len(segmented_audio)):
        #fbank_feat = logfbank(segmented_audio[i], 16000, nfilt=40)
        #fbank_feat -= (np.mean(fbank_feat, axis=0) + 1e-8)
        #fbank_feat /= (np.std(fbank_feat, axis=0) + 1e-8)
        
        # extract MFCC
        mfcc_feat = mfcc(segmented_audio[i], 16000)
        mfcc_feat = (mfcc_feat - np.mean(mfcc_feat, 0)) / (np.std(mfcc_feat, 0) + 1e-3)
        
        combined_features.append(mfcc_feat)
        combined_labels.append(segmented_labels[i])
        

    # save the features
    #combined_features = np.array(combined_features, dtype=object)
    #np.save(os.path.join("original_audio/segmented", split, filename + ".npy"), combined_features)
    
    # save the transcripts
    #segmented_labels = " ".join(segmented_labels)
    #with open(os.path.join("original_transcripts/combined_segmented/", split + "_segmented.txt"), "a") as f:
    #    f.write(segmented_labels.lower() + "\n")



def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def display_segment(i, segmented_audio, segmented_labels):
        print(waveform.size())
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        file_id = speech_file.split("/")[-1].replace(".flac", "")
        if "'" in word.label:
            word.label = word.label.replace("'", "+")
        
        segmented_audio.append(waveform[:, x0:x1])
        segmented_labels.append(word.label)

########################################################################################