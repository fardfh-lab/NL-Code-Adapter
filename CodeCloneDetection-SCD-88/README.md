# SCD-88


## Task Definition

Given a program and a set of candidates as inputs, the task aims to retrieve the top-k semantically similar codes.


## Evaluation Metric
MAP@R (Mean Average Precision) score: As the name suggests, it's the mean of the average precision scores obtained via contrasting the given query program against other candidate codes. Here precision is defined for retrieving the R most similar samples (R is the number of codes present in the same class, here R=129)

## Dataset

The [SCD-88](https://zenodo.org/record/5388452) dataset is the Python-specific subset of the [Cross-Language Clone Detection dataset](https://ieeexplore.ieee.org/document/8816761/) which was originally extracted from [AtCoder](https://atcoder.jp/), a popular Online Judge. To maintain consistency, We reformulate this classification task as a retrieval one. The programs/functions present in this dataset are in Python.

### Data Format

The lines within each of the three jsonline splits:`train.jsonl`, `valid.jsonl`, `test.jsonl`, represents a function and a row can be represented as:

   - **code:** the source code
   - **label:** the number of problem that the source code solves
   - **index:** the index of the example

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Problems | #Examples |
| ----- | --------- | :-------: |
| Train | 60        |   7800    |
| Dev   | 8         |   1040    |
| Test  | 20        |   2600    |

## CLI Usage

### Fine-tuning and Inference

`code/run_scd.py` contains a pipeline for fine-tuning BERT-based models on this task. You can specify the `--do_train` argument to fine-tune a pre-trained model and the `--do_test` argument to perform inference on test data. Consequently, you could follow our example to fine tune and use `microsoft/codebert-base` on the SCD-88 dataset.

```shell
export OUTPUT_DIR=/path/to/output/dir
export MODEL_NAME=/path/to/model

python code/run_scd.py \
		--output_dir $OUTPUT_DIR \
		--model_type "roberta" \
		--model_name_or_path $MODEL_NAME \
		--tokenizer_name "roberta-base" \
		--do_train \
		--do_eval \
		--do_test \
		--train_data_file dataset/train.jsonl \
		--eval_data_file dataset/valid.jsonl \
		--test_data_file dataset/test.jsonl \
		--epoch 2 \
		--block_size 425 \
		--train_batch_size 16 \
		--eval_batch_size 8 \
		--learning_rate 2e-5 \
		--max_grad_norm 1.0 \
		--weight_decay 0.01 \
		--evaluate_during_training \
		--seed 123456 2>&1| tee scd.log
```

 where
 - `$OUTPUT_DIR` is the name of the directory in which the trained model, task adapters and evaluation results are saved.
 - `$MODEL_NAME` is the name of a pretrained model (e.g., `roberta-base` or `microsoft/codebert-base`) or the path to that model.

Additionally, you can train and use task adapters by appending (and/or modifying) the following arguments in the aforementioned command.  When training task adapters, be sure to specify the `$MODEL_NAME` as `roberta-base` and freeze the model's weights by including the `--freeze_ptlm` argument.

```shell
		--model_name_or_path "roberta-base" \
		--freeze_ptlm \
		--train_adapter \
		--task_adap_name $TASK_ADAP_NAME \
		--task_adapter_config "pfeiffer" \
		--lang_adapter_name $LANG_ADAP_NAME \
		--lang_adapter_config "pfeiffer+inv" \
		--load_adapter $LANG_ADAP_PATH \
		--save_task_adap \
		--epoch 15 \
```

<sup>*Note*: We used the following parameters to train our task adapters, keeping their reduction factor at the default value of 16.</sup>


### Evaluation

Once inferred, the generated prediction sets should be saved into your output directory `$OUTPUT_DIR`. They should look something like the sample answers `evaluator/answers.jsonl`.

```b
{"index": "0", "answers": ["1", "2"]}
{"index": "1", "answers": ["0", "2"]}
{"index": "2", "answers": ["0", "1"]}
{"index": "4", "answers": ["3", "5"]}
{"index": "3", "answers": ["4", "5"]}
{"index": "5", "answers": ["4", "3"]}
```

You can extract test answers and use the evaluation script `evaluator/scd_evaluator.py` by executing the following:

```shell
python evaluator/scd_extract_answers.py -c "dataset/test.jsonl" -o "$OUTPUT_DIR/answers.jsonl" 
python evaluator/scd_evaluator.py -a "$OUTPUT_DIR/answers.jsonl"   -p "$OUTPUT_DIR/predictions.jsonl" 
```
<br>

You could also try evaluating the answers generated from the sample test set `evaluator/test.jsonl`. The results should resemble the following format:
```
{'MAP@R': 0.5833}
```

## Results

The results on the test set are shown as below:

| Method                                                             |    MAP@R     |
| ------------------------------------------------------------------ | :----------: |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)                    |    73.90     |
| MODE-X<sub>CN</sub>                                                | <u>75.65</u> |
| MODE-X<sub>CSN</sub>                                               | <u>75.65</u> |
| [CodeBERT<sub>MLM</sub>](https://arxiv.org/pdf/2002.08155.pdf)     |  **80.71**   |
| [CodeBERT<sub>MLM+RTD</sub>](https://arxiv.org/pdf/2002.08155.pdf) |    78.95     |

<sup>*Note*: On increasing the learning rate from 2e-5 to 5e-4, a MAP@R of 79 was recorded with MODE-X<sub>CN</sub>.</sup>

## References
```
@inproceedings{8816761,
  title={Cross-Language Clone Detection by Learning Over Abstract Syntax Trees},
  author={Perez, Daniel and Chiba, Shigeru},
  booktitle={2019 IEEE/ACM 16th International Conference on Mining Software Repositories (MSR)},
  year={2019},
  pages={518-528},
  doi={10.1109/MSR.2019.00078}
}
```

```
@article{DBLP:journals/corr/abs-2102-04664,
  title={CodeXGLUE: {A} Machine Learning Benchmark Dataset for Code Understanding and Generation},
  author={Lu, Shuai and Guo, Daya and Ren, Shuo and Huang, Junjie and Svyatkovskiy, Alexey and Blanco, Ambrosio and Clement, Colin B. and Drain, Dawn and Jiang, Daxin and Tang, Duyu and Li, Ge and Zhou, Lidong and Shou, Linjun and Zhou, Long and Tufano, Michele and Gong, Ming and Zhou, Ming and Duan, Nan and Sundaresan, Neel and Deng, Shao Kun and Fu, Shengyu and Liu, Shujie},
  journal={CoRR},
  volume={abs/2102.04664},
  year={2021}
}
```


