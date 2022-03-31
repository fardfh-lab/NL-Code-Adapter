## Language Adapter Training
Fine-tuning (or training from scratch) the adapter modules for language modeling on a text dataset for ALBERT, BERT, DistilBERT, RoBERTa ...
The adapters for each of the models are trained or fine-tuned using a masked language modeling (MLM) loss. We experimented with the Houlsby, Pfeiffer and Pfeiffer-Inv adapter configurations trained over a frozen RoBERTa model.
The results in the paper are based on the `RoBERTa + Pfeiffer-Inv` configuration. You can find more information about the difference between these adapter configurations in the [adapter-hub documentation](https://docs.adapterhub.ml).

<b>NOTE:</b> The script contained in this folder has been modified from [the example scripts provided by HuggingFace for the Transformers library](https://github.com/huggingface/transformers/tree/master/examples) to support training and fine-tuning of language adapters instead of full model fine-tuning.

Our script uses a custom training loop and leverages the 🤗 Accelerate library. 
We provide support to run training and validation both for datasets hosted on the 🤗 [hub](https://huggingface.co/datasets) or with your own text files.
You can easily customize them to your needs if you need extra processing on your datasets.

Before getting started with adapter training, make sure to have everything set up by following the instructions provided [here](https://github.com/fardfh-lab/NL-Code-Adapter).

## Datasets
We train and evaluate language adapters on two datasets: [CodeNet](https://developer.ibm.com/exchanges/data/all/project-codenet/) by IBM, and [CodeSearchNet](https://github.com/github/CodeSearchNet) which was a joint effort from GitHub and Microsoft Research. 
The details for language adapter training are provided below.

* <b>CodeNet</b>: A large scale, high quality data collected for studies of artificial intelligence for code. We train language adapters on the Python, C/C++ and Java benchmark datasets provided on the project's [homepage](https://developer.ibm.com/exchanges/data/all/project-codenet/). We randomly split the data into 90-10 splits for training and validation respectively.
* <b>CodeSearchNet</b>: The primary dataset consists of 2 million (comment, code) pairs from open source libraries. While the dataset consists of 6 programming languages, we train language adapters for the Python and Java benchmarks provided with the dataset. Summary statistics about this dataset can be found [in this notebook](https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb).

<b>NOTE:</b> Please refer to our research paper for more information on the preprocessing strategies used for each dataset.

## Language Adapters for RoBERTa and masked language modeling
The following example trains language adapters for the C/C++ benchmark provided by CodeNet. 
If you have followed all the instructions for the initial setup, running the following code will save the language adapters in the `storage` folder. 
As BERT/RoBERTa have a bidirectional mechanism; we're using the same loss that was used during their pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. 
The model may, therefore, converge slightly slower (over-fitting takes more epochs).

We trained the model on 4\**V100 GPUs*. You can also run the pre-training for language adapters on freely available GPUs on platforms like [*Google Colab*](https://colab.research.google.com/?utm_source=scs-index) by altering the `--per_device_batch_size` to an appropriate value.

```python3
export OUTPUT_DIR=PATH_TO_STORAGE
export TRAIN_FILE=PATH_TO_TRAIN_C++_CODENET
export VALIDATION_FILE=PATH_TO_VALID_C++_CODENET

python mlm_adap.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path 'roberta-base' \
    --tokenizer_name 'roberta-base' \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --adapter_config 'pfeiffer+inv' \
    --save_adap \
    --adap_name 'cpp_iso' \ 
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --max_seq_length 400 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --max_train_steps 150000 \
    --mlm_probability 0.15 \
    --train_adapter \
    --freeze_ptlm \  # Freeze RoBERTa layers
    --seed 123456 2>&1 | tee ../logs/languageAdapter.log
```
If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script concatenates all texts and then splits them in blocks of the same length).

You can also modify the script to train language adapters on a dataset hosted on the hub. For an insight on how to adapt our script for data from the hub, you can refer to the [original MLM script provided by HuggingFace](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling).

To fine-tune pre-trained language adapters on a custom dataset, you just need to change the value of the `--adap_name` flag to an appropriate language adapter hosted on [AdapterHub](https://adapterhub.ml). Please ensure that `--adapter_config` matches the language adapter configuration mentioned on the model card.

## Evaluating the Language Adapters
Running the training script with the `--do_eval` flag will print the perplexity scores of the model during language adapter pre-training after each epoch. While looking at the perplexity is a good way to ensure that adapter training with masked language modeling is headed in the right direction, perplexity is a weak measure to evaluate the overall performance of the LM. 

To probe deeper, we evaluate the trained language adapters on various *Cloze Style Tests* presented alongwith the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) benchmark. For implementation details and more information about the tests, please view the section on [cloze-testing](https://github.com/fardfh-lab/NL-Code-Adapter/tree/main/ClozeTest).

## Cite
If you use this code or our pre-trained language adapters, please consider citing the Paper and AdapterHub:

<pre><code>@inproceedings{pfeiffer2020AdapterHub,
    title={AdapterHub: A Framework for Adapting Transformers},
    author={Pfeiffer, Jonas and
            R{\"u}ckl{\'e}, Andreas and
            Poth, Clifton and
            Kamath, Aishwarya and
            Vuli{\'c}, Ivan and
            Ruder, Sebastian and
            Cho, Kyunghyun and
            Gurevych, Iryna},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    pages={46--54},
    year={2020}
}</code>
</pre>