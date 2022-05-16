"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd
import transformers
from transformers.integrations import is_ray_tune_available
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator
import copy

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.
training_args = transformers.TrainingArguments(output_dir='/scratch/yz5880/301',
                                               do_train=True,
                                               do_eval=True,
                                               evaluation_strategy='epoch',
                                               per_device_train_batch_size = 8,
                                               per_device_eval_batch_size=64,
                                               learning_rate=2e-5,
                                               weight_decay=0.01,
                                               num_train_epochs=3.0,
                                               logging_first_step=True,
                                               logging_steps=500,
                                               save_strategy='epoch',
                                               load_best_model_at_end=True)

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.
trainer = transformers.Trainer(
                               args=training_args,
                               train_dataset=train_data,
                               eval_dataset=val_data,
                               tokenizer=tokenizer,
                               model_init=finetuning_utils.model_init,
                               compute_metrics=finetuning_utils.compute_metrics,
)


def new_hp_space(trial):
     assert is_ray_tune_available()
     return {
         "num_train_epochs": 3,
         "training_batch_size": tune.grid_search([8, 32]),
         "learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5])
         }

def new_compute_objective(metrics):
    metrics = copy.deepcopy(metrics)
    acc = metrics.pop("accuracy", None)
    _ = metrics.pop("epoch", None)
     # Remove speed metrics
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return acc if len(metrics) == 0 else sum(metrics.values())

best_result = trainer.hyperparameter_search(hp_space=new_hp_space,
                                            compute_objective=new_compute_objective,
                                            n_trials=2,
                                            direction='maximize',
                                            backend='ray',
                                            search_alg=BasicVariantGenerator(),
                                            mode='max',
                                            log_to_file=True,
                                            local_dir='/scratch/yz5880/301'
                                            )
print(best_result)
