# BigCloneBench (BCB)

## Task Definition

Presented with two code fragments, this binary classification (0/1) task aims to identify semantic equivalence (1) between them.

## Evaluation Metric
Binary F1-Score

## Dataset

The [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) dataset was acquired from [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf) and filtered according to the [following paper](https://arxiv.org/pdf/2002.08653.pdf). This dataset contains Java code fragments.

### Data Format

1. `dataset/data.jsonl` contains the jsonline formatted dataset, each line representing one function.  One row can be illustrated as:

   - **func:** the function

   - **idx:** index of the example

2. The splits can be found under `dataset/train.txt`, `dataset/valid.txt`, and `dataset/test.txt` in the following format:    idx1	idx2	label

### Data Statistics

|       | #Examples |
| ----- | :-------: |
| Train |  901,028  |
| Dev   |  415,416  |
| Test  |  415,416  |



## CLI Usage

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- scikit-learn>=1.0.1

### Fine-tuning and Inference
`code/run_bcb.py` contains a pipeline for fine-tuning BERT-based models on this task. You can specify the `--do_train` argument to fine-tune a pre-trained model and the `--do_test` argument to perform inference on test data. Consequently, you could follow our example to fine tune and use `microsoft/codebert-base` on the BCB dataset.

```shell
export OUTPUT_DIR=/path/to/output/dir
export MODEL_NAME=/path/to/model

python code/run_bcb.py \
		--output_dir $OUTPUT_DIR \
		--model_type "roberta" \
		--model_name_or_path $MODEL_NAME \
		--tokenizer_name "roberta-base" \
		--do_train \
		--do_eval \
		--do_test \
		--train_data_file dataset/train.txt \
		--eval_data_file dataset/valid.txt \
		--test_data_file dataset/test.txt \
		--epoch 2 \
		--block_size 400 \
		--train_batch_size 16 \
		--eval_batch_size 32 \
		--learning_rate 5e-5 \
		--max_grad_norm 1.0 \
		--evaluate_during_training \
		--seed 123456 2>&1| tee bcb.log
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

Once inferred, the generated predictions should be saved into your output directory `$OUTPUT_DIR`. They should look something like the sample predictions `evaluator/predictions.txt`.

```b
13653451	21955002	0
1188160	8831513	1
1141235	14322332	0
16765164	17526811	1
```

You can use the evaluation script `evaluator/bcb_evaluator.py` by executing the following:

```shell
python evaluator/bcb_evaluator.py -a "dataset/test.txt" -p "$OUTPUT_DIR/predictions.txt"
```
<br>

You could also try running the evaluator on sample predictions present in `evaluator/predictions.txt`. The results should resemble the following format:
```
{'Recall': 0.25, 'Prediction': 0.5, 'F1': 0.3333333333333333}
```

## Results

The results on the test set are shown as below:

| Method                                                                          |      F1      |
| ------------------------------------------------------------------------------- | :----------: |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)                                 |    95.96     |
| MODE-X<sub>CN</sub>                                                             |    96.43     |
| MODE-X<sub>CSN</sub>                                                            | <u>96.61</u> |
| [CodeBERT<sub>MLM</sub>](https://arxiv.org/pdf/2002.08155.pdf)  (Reproduced)    |    96.38     |
| [CodeBERT<sub>MLM+RTD</sub>](https://arxiv.org/pdf/2002.08155.pdf) (Reproduced) |  **96.65**   |

## References
```
@inproceedings{svajlenko2014towards,
  title={Towards a big data curated benchmark of inter-project code clones},
  author={Svajlenko, Jeffrey and Islam, Judith F and Keivanloo, Iman and Roy, Chanchal K and Mia, Mohammad Mamun},
  booktitle={2014 IEEE International Conference on Software Maintenance and Evolution},
  pages={476--480},
  year={2014},
  organization={IEEE}
}
```

```
@inproceedings{wang2020detecting,
  title={Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree},
  author={Wang, Wenhan and Li, Ge and Ma, Bo and Xia, Xin and Jin, Zhi},
  booktitle={2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  pages={261--271},
  year={2020},
  organization={IEEE}
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
