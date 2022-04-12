import json
import os
import random
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess_data(file_loc='code_dataset.jsonl', generate_histogram=False):
    """
    Loads and processing the jsonl file,

    :param file_loc: location of target jsonl file
    :param generate_histogram: Flag to display histogram of function lengths
    :return: dataframe of preprocessed jsons
    """

    with open(file_loc, 'r') as json_file:
        json_list = list(json_file)

    code_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        code_list.append(result)

    code_df = pd.DataFrame(code_list)

    total = code_df['target'].sum()
    proportion = total / code_df.shape[0]

    print("Insecure code counts: {}, Total code counts: {}, Proportion {}".format(total, code_df.shape[0], proportion))

    if generate_histogram:
        plt.hist(code_df['func'].str.len(), bins=100)
        plt.show()

    return code_df


def split_data(input_data, attention_data, label_data, train_ratio=0.8, val_ratio=0.10, max_len=512):
    """
    Splits data in accordance with provdied ratios, additionally discards functions with > max_len tokens
        as these will not be processed by the model will (can truncate, yet may truncate the error in the code)

    :param input_data: input functions
    :param attention_data: attention map
    :param label_data: target labels
    :param train_ratio: ratio of data to train on
    :param val_ratio: ratio of data to validate with (test is inferred from this and train)
    :param max_len: max number of tokens allowed for training date

    :return: 3 tuples for train val and test containing (input, attention, target)
    """
    # Removing excessively long elements from dataset
    valid_token_index = [i for i in range(len(input_data)) if len(input_data[i]) <= max_len]
    X_data = np.array(input_data)[valid_token_index]
    A_data = np.array(attention_data)[valid_token_index]
    Y_data = np.array(label_data)[valid_token_index]

    dataset_size = len(X_data)

    # Determining index to split dataset
    random_id = random.sample(range(dataset_size), dataset_size)
    train_split_id = int(train_ratio * dataset_size)
    val_split_id = int((train_ratio + val_ratio) * dataset_size)

    train_ids = random_id[:train_split_id]
    val_ids = random_id[train_split_id:val_split_id]
    test_ids = random_id[val_split_id:]

    X_train = torch.tensor(list(X_data[train_ids]))
    A_train = torch.tensor(list(A_data[train_ids]))
    Y_train = torch.tensor(list(Y_data[train_ids]))

    X_val = torch.tensor(list(X_data[val_ids]))
    A_val = torch.tensor(list(A_data[val_ids]))
    Y_val = torch.tensor(list(Y_data[val_ids]))

    X_test = torch.tensor(list(X_data[test_ids]))
    A_test = torch.tensor(list(A_data[test_ids]))
    Y_test = torch.tensor(list(Y_data[test_ids]))

    return (X_train, A_train, Y_train), (X_val, A_val, Y_val), (X_test, A_test, Y_test)


def tokenize(code_df, model_name='codebert-base'):
    """
    Apply the tokenizer from the huggingface pretrained model

    :param code_df: dataframe of preprocess code (from jsonl)
    :param model_name: model name (targeting local install)
    :return: 3 tuples for train val and test containing (input, attention, target)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(code_df['func'].tolist(), truncation=False, padding='max_length')

    input_data = inputs['input_ids']
    attention_data = inputs['attention_mask']
    label_data = torch.tensor(code_df['target'].tolist())  # TODO - this can be directly converted to a np array

    return split_data(input_data, attention_data, label_data, max_len=512)



def train(model, train_data, val_data, epochs=5, batch_size=16, learning_rate=2e-5, validate_per=500,
          run_name="temp", run_descrption=None):
    """
    Main fine-tuning training loop for the provided model

    :param model: model loaded with predefined weights
    :param train_data: tuple of X_train, A_train, Y_train (X = inputs, A = attention, Y = target)
    :param val_data: tuple X_val, A_val, Y_val
    :param epochs: number of epochs for training
    :param batch_size: batch size (see note below about batch_hack)
    :param learning_rate: optimizer learning rate
    :param validate_per: number of weight updates before validation occurs
                            (notes: - if batch_size = 32, and validate_per = 32, validation will occur every batch
                                    - this is wrt the start of each epoch
                                    - validation will always occour at the start of each epoch (step 0))
    :param run_name: name used to saving checkpoints and log files within codebert_finetune_runs
    :param run_descrption: string that is saved to info.txt describing the run


    :return: None (models are saved in checkpoints along with log data)
    """

    # Creating dir to save logs and checkpoints, re
    dir_name = "codebert_finetune_runs/{}".format(run_name)
    if os.path.exists(dir_name):
        print("run name already exists, exiting to prevent overwriting")
        return 0
    else:
        os.makedirs(dir_name)

    # Saving run description.txt
    if run_descrption is not None:
        with open("{}/info.txt".format(dir_name), "a+") as f:
            f.write(run_descrption)

    # Unpacking data
    X_train, A_train, Y_train = train_data
    X_val, A_val, Y_val = val_data

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    batch_hack = batch_size  # See note below regarding limited GPU memory

    # Initializing arrays for tracking loss
    train_loss_hist = []
    val_loss_hist = []

    # Counter to track batches (see note below related to GPU memory)
    batch_count = 0
    # validate_per_batch = int(validate_per/batch_hack)

    # Moving model to GPU if configured
    model = model.to(device)
    for epoch in range(epochs):

        # Generating random index for manual shuffling of data each epoch as note using DataLoaders
        permutation = torch.randperm(X_train.shape[0])

        # Note here that only a single element is loaded at each iteration (batch size = 1) due to GPU memory constraint
        for i in range(0, X_train.shape[0], 1):

            # Loading batch and moving to device
            indices = permutation[i:i + 1]
            batch_X, batch_Y = X_train[indices].to(device), Y_train[indices].to(device), \

            model.train()

            # Forward pass
            outputs = model(batch_X, labels=batch_Y)

            # Backward pass
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Tracking loss
            train_loss_hist.append(loss.item())

            # Training output
            train_output = "Epoch:{} Step:{} Training_loss:{:.6f}".format(epoch, i, loss.item())
            print(train_output+" Training_loss_avg:{:.6f}".format(np.average(train_loss_hist[-50:])))
            with open("{}/train_loss.txt".format(dir_name), "a+") as f:
                f.write(train_output+"\n")

            # Validation
            if (i+1) % validate_per == 0:
                val_loss_total = 0
                model.eval()
                print("Validating:")
                for j in tqdm(range(0, X_val.shape[0], 1)):
                    # Loading singular validation data (overwrites train data as can only load 1 intp GPU)
                    batch_X, batch_Y = X_val[j].to(device).reshape(1, -1), Y_val[j].to(device).reshape(1, -1)
                    with torch.no_grad():
                        val_outputs = model(batch_X, labels=batch_Y)
                    val_loss_total += val_outputs['loss'].item()

                # Adding average loss to tracker
                val_average = val_loss_total / (X_val.shape[0])
                val_loss_hist.append(val_average)

                # Validation output and logging
                val_output = "Epoch:{} Step:{} Val_loss:{:.6f}".format(epoch, i, val_average)
                print(val_output)
                with open("{}/val_los.txt".format(dir_name), "a+") as f:
                    f.write(val_output+"\n")

        # End of epoch checkpoint
        model.save_pretrained("{}/epoch_{}".format(dir_name, epoch + 1))


def main():
    """
    Main configuration function for a given finetune run
    :return: None
    """

    run_name = "test2"
    model_name = 'codebert-base'
    checkpoint_location = None

    code_df = preprocess_data(file_loc='code_dataset.jsonl')
    train_data, val_data, test_data = tokenize(code_df, model_name=model_name)

    # Loading model from checkpoint if location provided
    if checkpoint_location is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_location)

    train(model=model,
          train_data=train_data,
          val_data=val_data,
          epochs=5,
          batch_size=1,
          learning_rate=1e-4,
          validate_per=500,
          run_name=run_name,
          run_descrption="Debugging, lr=2e-5, validate per 500, batch 16")


if __name__ == "__main__":
    main()
