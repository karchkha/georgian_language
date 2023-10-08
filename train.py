


import sys

if len(sys.argv) > 1:
    arg = sys.argv[1]



# PROJECT_NAME = "cv-fpdr01"
# PROJECT_NAME = "test"
PROJECT_NAME = arg

##################### import data and also print resulsts ###################
from datasets import load_dataset


common_voice_train = load_dataset('data_utils/common_voice.py', "ka", split="validated+other[:40%]")
common_voice_validation = load_dataset('data_utils/common_voice.py', "ka", split="other[40%:70%]")
common_voice_test = load_dataset('data_utils/common_voice.py', "ka", split="other[70%:]")

print(common_voice_train)
print(common_voice_validation)


################################ processing data ##############################################

from transformers import Wav2Vec2Processor
from data_utils.dataloader import CommonVoiceDataset, DataCollatorCTCWithPadding

processor = Wav2Vec2Processor.from_pretrained("processor")

# tokenized_datasets = tokenized_datasets.remove_columns(books_dataset["train"].column_names)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

train_dataset = CommonVoiceDataset(common_voice_train, processor=processor, column_names=["input_values", "labels"], downsample = False)
val_dataset = CommonVoiceDataset(common_voice_validation, processor=processor, column_names=["input_values", "labels"], downsample = False)
# test_dataset = CommonVoiceDataset(common_voice_test, processor=processor, column_names=["input_values", "labels"])

######################################### Metrics ##############################################


from datasets import load_dataset, load_metric
import numpy as np

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    

############################ training parameters ###########################

from transformers import Wav2Vec2ForCTC
from transformers.trainer_utils import get_last_checkpoint

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-1b", #"facebook/wav2vec2-large-xlsr-53", #, #if not last_checkpoint else last_checkpoint, 
    attention_dropout=0.1,    # 0.1
    hidden_dropout=0.1,       # 0.1
    feat_proj_dropout=0.1,    # 0.0  
    mask_time_prob=0.05,
    layerdrop=0.1,            # 0.1  
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=processor.tokenizer.vocab_size,
    cache_dir = "./" ### this is the original pretrained model
)


# frezzee encoder!!!! 

model.freeze_feature_extractor() #freeze_feature_encoder


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="logs/" + PROJECT_NAME,
    group_by_length=True,
    per_device_train_batch_size=2, #30,  #16
    gradient_accumulation_steps=2,
    evaluation_strategy="steps", #"epoch", #
    gradient_checkpointing=True,
    # save_strategy = "epoch", 
    num_train_epochs=50, # Just for demo, change it
    fp16=True,
    save_steps=2000, # Just for demo, change it
    eval_steps=2000, # Just for demo, change it
    logging_steps=2000, # Just for demo, change it
    learning_rate= 4.5e-5, #3e-4,
    warmup_steps =500, # Just for demo, change it
    # save_total_limit=2,
)


#########################  define trainer ################################

from trainer import CommonVoiceTrainer 

trainer = CommonVoiceTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
)


##################### chack for existing checkpoints #################

import os
from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = None

save_dir = "logs/" + PROJECT_NAME

if os.path.exists(save_dir):
    last_checkpoint = get_last_checkpoint(save_dir)
    
print(last_checkpoint if last_checkpoint else 0)


#################### TRAIN #########################

if last_checkpoint:
    print(f"last_checkpoint: {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    train_result = trainer.train()