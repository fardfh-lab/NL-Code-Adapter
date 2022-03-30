# POJ-104


## Task Definition

Given a program and a set of candidates as inputs, the task aims to retrieve the top-k semantically similar codes.

## Evaluation Metric
**MAP@R (Mean Average Precision) score:** As the name suggests, it's the mean of the average precision scores obtained via contrasting the given query program against candidate codes. Here precision is defined for retrieving the R most similar samples (R is the number of codes present in the same class, here R=499)

## Dataset

The [POJ-104](https://arxiv.org/pdf/1409.5718.pdf) dataset (acquired from [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf)). The programs/functions present here are in C/C++.

### Download and Preprocess

1. Download and extract the [dataset](https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?usp=sharing) into the `dataset/` directory or run the following command:

```shell
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
tar -xvf programs.tar.gz
```

2. Once downloaded, it can be preprocessed via `dataset/preprocess.py`.

```shell
python preprocess.py
cd ..
```

### Data Format

Once preprocessed, you can find three the jsonline splits within the `dataset/` directory: `train.jsonl`, `valid.jsonl`, `test.jsonl`. Each line within the files represents a function and a row can be represented as:

   - **code:** the source code
   - **label:** the number of problem that the source code solves
   - **index:** the index of the example

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Problems | #Examples |
| ----- | --------- | :-------: |
| Train | 64        |  32,000   |
| Dev   | 16        |   8,000   |
| Test  | 24        |  12,000   |

## CLI Usage

### Fine-tuning and Inference

`code/run_poj.py` contains a pipeline for fine-tuning BERT-based models on this task. You can specify the `--do_train` argument to fine-tune a pre-trained model and the `--do_test` argument to perform inference on test data. Consequently, you could follow our example to fine tune and use `microsoft/codebert-base` on the POJ-104 dataset.

```shell
export OUTPUT_DIR=/path/to/output/dir
export MODEL_NAME=/path/to/model

python code/run_poj.py \
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
		--block_size 400 \
		--train_batch_size 8 \
		--eval_batch_size 16 \
		--learning_rate 2e-5 \
		--max_grad_norm 1.0 \
		--evaluate_during_training \
		--seed 123456 2>&1| tee poj.log
```

 where
 - `$OUTPUT_DIR` is the name of the directory in which the trained model, task adapters and evaluation results are saved.
 - `$MODEL_NAME` is the name of a pretrained model (e.g., `roberta-base` or `microsoft/codebert-base`) or the path to that model.

Additionally, you can train and use task adapters by appending (and/or modifying) the following arguments in the aforementioned command.  When training task adapters, be sure to specify the `$MODEL_NAME` as `roberta-base` and freeze the model's weights by including the `--freeze_ptlm` argument. Further, We drop the last two layers as it improves performance on BCB. In order to do so, simply pass the list of layers (integer values prefixed with 'l') you want to drop through the `--drop_layers` argument.

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
		--drop_layers 'l10' 'l11' \
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

You can extract test answers and use the evaluation script `evaluator/poj_evaluator.py` by executing the following:

```shell
python evaluator/poj_extract_answers.py -c "dataset/test.jsonl" -o "$OUTPUT_DIR/answers.jsonl" 
python evaluator/poj_evaluator.py -a "$OUTPUT_DIR/answers.jsonl"   -p "$OUTPUT_DIR/predictions.jsonl" 
```
<br>

You could also try evaluating the answers generated from the sample test set `evaluator/test.jsonl`. The results should resemble the following format:
```
{'MAP@R': 0.5833}
```

## Results

The results on the test set are shown as below:

| Method                                                                          |    MAP@R     |
| ------------------------------------------------------------------------------- | :----------: |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)                                 |    81.52     |
| MODE-X<sub>CN</sub>                                                             | <u>82.40</u> |
| [CodeBERT<sub>MLM</sub>](https://arxiv.org/pdf/2002.08155.pdf)  (Reproduced)    |    85.08     |
| [CodeBERT<sub>MLM+RTD</sub>](https://arxiv.org/pdf/2002.08155.pdf) (Reproduced) |  **86.48**   |

## References
```
@inproceedings{mou2016convolutional,
  title={Convolutional neural networks over tree structures for programming language processing},
  author={Mou, Lili and Li, Ge and Zhang, Lu and Wang, Tao and Jin, Zhi},
  booktitle={Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence},
  pages={1287--1293},
  year={2016}
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


