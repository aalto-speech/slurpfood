import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

import forced_alignment

from captum.attr import IntegratedGradients, TokenReferenceBase, FeatureAblation

from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm
import librosa
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torchaudio


class SER(sb.Brain):
    def compute_forward(self, sig, lens):
        #"Given an input batch it computes the output probabilities."        
        outputs = self.modules.wav2vec2(sig)
        # take a specific layer
        outputs = outputs[-1]

        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        # apply label linear
        outputs = self.modules.label_lin(outputs)
        outputs = self.hparams.log_softmax(outputs)

        return outputs 


    def plotting(sig, i, segmented_audio, segmented_labels, averaged_attributions, words, scenario_label, action, important_word):
        # Plot the signal
        sample_rate = 16000  # Sample rate of your audio (replace with the actual sample rate)
        duration = sig.size(1) / sample_rate  # Duration of your audio in seconds (replace with the actual duration)
        num_samples = int(sample_rate * duration)

        # Prepare data for plotting
        viz_data = []
        for j in range(len(segmented_audio)):
            viz_data.append({"word": segmented_labels[j], "start_time": segmented_audio[j][0], "end_time": segmented_audio[j][1]})

        # plot the signal
        viz_sig = sig[0].cpu().detach().numpy()
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)

        plt.plot(viz_sig)

        # Plot transcript segments
        for entry in viz_data:
            start_sample = int(entry["start_time"] * sample_rate)
            end_sample = int(entry["end_time"] * sample_rate)
            segment = viz_sig[start_sample:end_sample]
            plt.fill_betweenx(y=[min(viz_sig), max(viz_sig)],
                            x1=entry["start_time"],
                            x2=entry["end_time"],
                            color="lightblue", alpha=0.5, label=entry["word"])
        plt.legend()
        ## Done ####

        # Plot the heatmap
        plt.subplot(2, 1, 2)
        sns.heatmap(averaged_attributions, cmap='viridis', xticklabels=words, cbar=True)
        # Done
        
        # create directory with the name of the scenario if it does not exist
        if not os.path.exists(os.path.join("outputs", scenario_label)):
            os.makedirs(os.path.join("outputs", scenario_label))

        # save the plot
        plt.tight_layout()
        plt.savefig(os.path.join("outputs", scenario_label, action + "_" + important_word + "_" + str(i) + ".png"))

    
    def compute_ig(
            self,
            dataset, # Must be obtained from the dataio_function
            min_key, # We load the model with the lowest error rate
            loader_kwargs, # opts for the dataloading
            id2trn,
            aligner_model,
            dictionary,
            idx2label
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)
        
        fa = FeatureAblation(self.compute_forward)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                try:
                    batch = batch.to(device)
                    sig, lens = batch.sig
                    label, _ = batch.labels_encoded
                    action = batch.action

                    # get the transcript associated with the utterance
                    transcript = id2trn[batch.id[0]]

                    # pass through forward to get probabilities
                    probs = self.compute_forward(sig, lens)
                    # get the element with highest probability
                    topi, topk = probs.topk(1)
                    topk = topk.item()

                    # USE THE DATA ONLY FOR THE WRONG OR CORRECT PREDICTIONS
                    # if topk != labels.item():
                    # ALIGNMENT
                    segmented_audio, segmented_labels = alignment(aligner_model, dictionary, sig, transcript, device)

                    words = []
                    # Create masks
                    sig_mask = torch.zeros(sig.size(1))
                    num_words = 0
                    for start_time, end_time in segmented_audio:
                        num_words += 1
                        sig_mask[start_time:end_time] = num_words
                    sig_mask = sig_mask.unsqueeze(0)
                    # convert to int tensor
                    sig_mask = sig_mask.type(torch.LongTensor).to(device)

                    # COMPUTE FA
                    attributions_fa = fa.attribute(sig, target=topk, feature_mask=sig_mask, additional_forward_args=(lens))

                    # Aggretae the attributions based on the words
                    averaged_attributions = []
                    for segment in range(num_words):
                        averaged_attributions.append(torch.mean(attributions_fa[sig_mask == segment]))
                    averaged_attributions = torch.stack(averaged_attributions)
                    averaged_attributions = averaged_attributions.cpu().detach().numpy()
                    # DONE COMPUTING FA

                    # Get the scenario, intent, and most important word labels
                    scenario_label = idx2label[label.item()]
                    action = action[0]
                    # get the index with the highest averaged_attributions value
                    # max_idx = np.argmax(np.abs(averaged_attributions))
                    max_idx = np.argmax(averaged_attributions)
                    important_word = segmented_labels[max_idx]
                
                    # ### PLOTTING ####
                    # #self.plotting(sig, i, segmented_audio, segmented_labels, averaged_attributions, words, scenario_label, action, important_word)
                    # # DONE PLOTTING

                    # save the id, trn, scenario, action, and important word in a CSV
                    with open("outputs/oov/ablation/most_important_words_pred_class_all_preds.csv", "a") as f:
                            f.write(batch.id[0] + "," + transcript + "," + scenario_label + "," + action + "," + important_word + "\n")
                except Exception as e:
                    print(e)


def alignment(aligner_model, dictionary, sig, transcript, device):
    # do the alignment
    transcript = transcript.replace(" ", "|")
    transcript = transcript.replace("@", "")
    transcript = transcript.replace("#", "")
    transcript = transcript.replace(".", "")
    transcript = transcript.upper()

    with torch.inference_mode():
        # waveform, _ = torchaudio.load(data_path)
        emissions, _ = aligner_model(sig.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)
    
    emission = emissions[0].cpu().detach()
    tokens = [dictionary[c] for c in transcript]
    trellis = forced_alignment.get_trellis(emission, tokens)
    
    path = forced_alignment.backtrack(trellis, emission, tokens)
    segments = forced_alignment.merge_repeats(path, transcript)
    word_segments = forced_alignment.merge_words(segments)
    
    segmented_audio = []
    segmented_labels = []
    for seg in range(len(word_segments)):
        ratio = sig.size(1) / (trellis.size(0) - 1)
        word = word_segments[seg]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        if "'" in word.label:
            word.label = word.label.replace("'", "+")
        
        segmented_audio.append([x0, x1])
        segmented_labels.append(word.label)

    return segmented_audio, segmented_labels



def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train_oov.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev_oov.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("data_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(data_path):
        # get current directory
        cwd = os.getcwd()
        file_id = data_path.split("/")[-1]
        data_path = os.path.join(cwd, "../../../slurp/scripts/audio/slurp_real", file_id)
        sig = sb.dataio.dataio.read_audio(data_path)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("scenario", "action")
    @sb.utils.data_pipeline.provides("labels_encoded")
    def text_pipeline(scenario, action):
        labels_encoded = hparams["label_encoder"].encode_sequence_torch([scenario])
        yield labels_encoded
        yield action

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    # load the encoder
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "labels_encoded", "action"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    # CuDNN with RNN doesn't support gradient computation in eval mode that's why we need to disable cudnn for RNN in eval mode
    torch.backends.cudnn.enabled=False
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # load the alignment model
    # load the model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    aligner_model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    dictionary = {c: i for i, c in enumerate(labels)}

    # create a label mapping
    # Check that to the oov and non-oov splits the mapping is the same
    idx2label = {
        0: "calendar",
        1: "audio",
        2: "weather",
        3: "lists",
        4: "email",
        5: "alarm",
        6: "play",
        7: "recommendation",
        8: "social",
        9: "news",
        10: "general",
        11: "qa",
        12: "iot",
        13: "transport",
        14: "cooking",
        15: "music",
        16: "datetime",
        17: "takeaway"
    }


    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Trainer initialization
    ser_brain = SER(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

    # Dataset creation
    train_data, valid_data, test_data = data_prep(hparams["path_to_data"], hparams)

    ## Get id2trn files
    with open(hparams["path_to_data"] + "/train_oov.json", "r") as f:
        train_meta = json.load(f)
    with open(hparams["path_to_data"] + "/dev_oov.json", "r") as f:
        dev_meta = json.load(f)
    with open(hparams["path_to_data"] + "/test.json", "r") as f:
        test_meta = json.load(f)

    id2trn = {}
    for elem in train_meta:
        id2trn[elem] = train_meta[elem]["transcript"].rstrip()
    for elem in dev_meta:
        id2trn[elem] = dev_meta[elem]["transcript"].rstrip()
    for elem in test_meta:
        id2trn[elem] = test_meta[elem]["transcript"].rstrip()


    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ser_brain.fit(
            ser_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # Evaluate
        print("Evaluating")
        ser_brain.compute_ig(test_data, "error_rate", hparams["test_dataloader_options"], id2trn, aligner_model, dictionary, idx2label)


if __name__ == "__main__":
    main()
