#  Cloze Style Testing for Language Adapters
We evaluate the performance of our language adapters on two kinds of cloze tests: ClozeTest-maxmin and ClozeTest-all. The tasks were introduced along with the [CodeXGLUE benchmark by Microsoft Research](https://github.com/microsoft/CodeXGLUE).

## Task Description
Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for a masked token given the context of the mask, which can be formulated as a multi-choice classification problem.

The language adapters were evaluated on cloze tests in the code domain for six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.

ClozeTest-maxmin and ClozeTest-all differ in their selected word sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words. Furthermore, ClozeTest-maxmin evaluates the language model's ability to understand code semantics while ClozeTest-all evaluates the language model's ability to learn syntactic representations from code.

<b>NOTE:</b> Please refer to our paper for a detailed discussion over the differences between the two tasks.

## Dependencies
* python 3.6 or 3.7
* torch==1.5.0
* adapter-transformers>=4.8.2

## Data
The data for cloze testing are collected from the the validation and test sets of CodeSearchNet, which contains six different programming languages including RUBY, JAVASCRIPT, GO, PYTHON, JAVA and PHP. Each instance contains a masked code function, its docstring and the target word. 

The preprocessed data for cloze testing is made available in the `data/cloze-<style>` directories.

Data statistics of ClozeTest-maxmin and ClozeTest-all are shown in the below table:

TASK | RUBY | JAVASCRIPT |  GO  | PYTHON | JAVA | PHP  | ALL  |
:-------: | :--: | :--------: | :--: | :----: | :--: | :--: | :--: |
CT-maxmin |  38  |    272     | 152  |  1264  | 482  | 407  | 2615 |
CT-all |  4437  |    13837     | 25282  |  40137  | 40492  | 51930  | 176115 |

## Run ClozeTest

You can run cloze tests for language adapters by the following command. It will automatically generate predictions to `--output_dir`.
We note that the best performing final language model is obtained upon dropping the adapters in the final or final and pre-final layers. To replicate our results, set `--drop_layers` to the list of adapter layers you want to drop from the resultant language model. Here we have presented an example experiment configuration for dropping the adapters in the pre-final and final layers.

```python
export MODE='maxmin'

export MODEL_NAME='RoBERTa'
export MODEL_CONFIG='roberta-base'

export DROP_FROM='l10' 

export ADAPTER_NAME='python_iso' # Ensure this matches the name of the pre-trained language adapter
export LANG_ADAPTER=PATH_TO_TRAINED_LANGUAGE_ADAPTERS

python code/run_cloze.py \
    --cloze_mode $MODE \
    --model $MODEL_CONFIG \
    --load_adapter $LANG_ADAPTER \
    --adapter_name $ADAPTER_NAME \
    --adapter_config 'pfeiffer+inv' \ # Ensure that this matches with the configuration of pre-trained language adapters
    --drop_layers 'l10' 'l11' \ # List of all adapters above DROP_FROM
    --langs "python" "java" "ruby" "go" "php" "javascript" \
    --output_dir "evaluator/predictions-$MODE/$MODEL_NAME/${DROP_FROM}/" | tee "evaluator/predictions-$MODE/$MODEL_NAME/logs/$MODE-${DROP_FROM}.log"
```

## Evaluator
We provide a script to evaluate predictions for the cloze tests, and report accuracy for the tasks. You can run evaluation by the following command:

```python
python evaluator/evaluator.py \
    --cloze_mode $MODE \
    --langs "python" "java" "ruby" "go" "php" "javascript" \
    --answers "evaluator/answers-$MODE/" \
    --predictions "evaluator/predictions-$MODE/$MODEL_NAME/${DROP_FROM}/" | tee -a "evaluator/predictions-$MODE/$MODEL_NAME/logs/$MODE-${DROP_FROM}.log"
```

## Result

Please refer to our paper for the final accuracy plots of the adapter drop experiments along with a detailed analysis of the Cloze Style tests using our approach.

## Cite
The cloze tests are built upon the CodeSearchNet dataset. If you use this code or the datasets, please consider citing Our Paper, CodeXGLUE, CodeBERT and CodeSearchNet:

<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code>
</pre>

<pre>
<code>@article{feng2020codebert,
  title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
  author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
  journal={arXiv preprint arXiv:2002.08155},
  year={2020}
}</code>
</pre>

<pre>
<code>@article{husain2019codesearchnet,
  title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}</code> 
</pre>


