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

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm
from confidence_intervals import evaluate_with_conf_int


class SER(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        sig_1, lens_1 = batch.sig_1
        sig_2, lens_2 = batch.sig_2

        outputs_1 = self.modules.wav2vec2(sig_1)
        outputs_2 = self.modules.wav2vec2(sig_2)
        # take a specific layer
        outputs_1 = outputs_1[-1]
        outputs_2 = outputs_2[-1]

        outputs_1 = self.hparams.avg_pool(outputs_1, lens_1)
        outputs_1 = outputs_1.view(outputs_1.shape[0], -1)
        outputs_2 = self.hparams.avg_pool(outputs_2, lens_2)
        outputs_2 = outputs_2.view(outputs_2.shape[0], -1)

        # take the average along the second dimension
        outputs = torch.mean(torch.stack([outputs_1, outputs_2]), dim=0)

        # apply label linear
        outputs = self.modules.label_lin(outputs)
        outputs = self.hparams.log_softmax(outputs)

        return outputs 


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label"""
        label, label_lens = batch.labels_encoded

        """to meet the input form of nll loss"""
        label = label.squeeze(1)
        loss = self.hparams.compute_cost(predictions, label)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, label)

        return loss


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
    

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )
        
        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
    
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    
    def run_inference(
            self,
            dataset, # Must be obtained from the dataio_function
            min_key, # We load the model with the lowest error rate
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            true_labels = []
            pred_labels = []
            for batch in dataset:
                output = self.compute_forward(batch, stage=sb.Stage.TEST) 
                
                topi, topk = output.topk(1)
                topk = topk.squeeze()

                labels, label_lens = batch.labels_encoded
                labels = labels.squeeze()

                topk = topk.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                
                try: 
                    for elem in labels:
                        true_labels.append(elem)

                    for elem in topk:
                        pred_labels.append(elem)
                except:
                    true_labels.append(labels)
                    pred_labels.append(topk)
                
            true_labels = np.array(true_labels)
            pred_labels = np.array(pred_labels)

            print("F1 score: ", round(f1_score(true_labels, pred_labels, average="micro"), 4) * 100)
            # Calculate confidence intervals
            print(evaluate_with_conf_int(pred_labels, f1_score, true_labels, conditions=None, num_bootstraps=1000, alpha=5))


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train_cg.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev_cg.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("data_path_1", "data_path_2")
    @sb.utils.data_pipeline.provides("sig_1", "sig_2")
    def audio_pipeline(data_path_1, data_path_2):
        cwd = os.getcwd()
        file_id_1 = data_path_1.split("/")[-1]
        file_id_2 = data_path_2.split("/")[-1]
        data_path_1 = os.path.join(cwd, "../../../slurp/scripts/audio/slurp_real", file_id_1)
        data_path_2 = os.path.join(cwd, "../../../slurp/scripts/audio/slurp_real", file_id_2)

        sig_1 = sb.dataio.dataio.read_audio(data_path_1)
        sig_2 = sb.dataio.dataio.read_audio(data_path_2)
        yield sig_1
        yield sig_2

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("scenario_1")
    @sb.utils.data_pipeline.provides("labels_encoded")
    def text_pipeline(scenario_1):
        labels_encoded = hparams["label_encoder"].encode_sequence_torch([scenario_1])
        yield labels_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["label_encoder"].update_from_didataset(train_data, output_key="scenario_1")
    hparams["label_encoder"].update_from_didataset(valid_data, output_key="scenario_1")
    hparams["label_encoder"].update_from_didataset(test_data, output_key="scenario_1")

    # save the encoder
    hparams["label_encoder"].save(hparams["label_encoder_file"])
    
    # load the encoder
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig_1", "sig_2", "labels_encoded"])
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
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
        # evaluate
        print("Evaluating")
        ser_brain.run_inference(test_data, "error_rate", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
