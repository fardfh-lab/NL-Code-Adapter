# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from tqdm import tqdm
from transformers import (
    RobertaConfig, 
    RobertaForMaskedLM, 
    RobertaTokenizer,
    AdapterConfig
)
import argparse
import json
import os

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)}


def get_cloze_words(filename, tokenizer):
    with open(filename, 'r', encoding='utf-8') as fp:
        words = fp.read().split('\n')
    idx2word = {tokenizer.encoder[w]: w for w in words}
    return idx2word


def test_single(text, model, idx2word, tokenizer, device):
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[:510]
    inputs = tokenizer.build_inputs_with_special_tokens(tokenized_text)
    index = inputs.index(tokenizer.mask_token_id)

    inputs = torch.tensor([inputs])
    inputs = inputs.to(device)

    with torch.no_grad():
        scores = model(inputs)[0]
        score_list = scores[0][index]
        word_index = torch.LongTensor(list(idx2word.keys())).to(device)
        word_index = torch.zeros(score_list.shape[0]).to(device).scatter(0, word_index, 1)
        score_list = score_list + (1-word_index) * -1e6
        predict_word_id = torch.argmax(score_list).data.tolist()

    return predict_word_id


def cloze_test(args, lang, model, tokenizer, device):
    cloze_words_file = os.path.join('data', 'cloze-'+args.cloze_mode, 'cloze_test_words.txt')
    file_path = os.path.join('data', 'cloze-'+args.cloze_mode, lang, 'clozeTest.json')

    idx2word = get_cloze_words(cloze_words_file, tokenizer)
    lines = json.load(open(file_path))

    results = []
    for line in tqdm(lines):
        text = ' '.join(line['nl_tokens'] + line['pl_tokens']) if args.use_comments else ' '.join(line['pl_tokens'])  
        predict_id = test_single(text, model, idx2word, tokenizer, device)
        results.append({'idx': line['idx'],
                        'prediction': idx2word[predict_id]})
    with open(os.path.join(args.output_dir, lang, 'predictions.txt'), 'w', encoding='utf-8') as fp:
        for inst in results:
            fp.write(inst['idx']+'<CODESPLIT>'+inst['prediction']+'\n')
    print(f"cloze-{args.cloze_mode} for {lang} finished")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        default='roberta-base', 
        help='"roberta-base" or "microsoft/codebert-base-mlm" or model path(pytorch_model.bin)'
    )
    parser.add_argument(
        '--load_adapter', 
        default=None, 
        help='Path to pre-trained language adapter.'
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="lang",
        help="Pass the name of the language adapter to be included"
    )
    parser.add_argument(
        '--adapter_config',
        type=str,
        default='pfeiffer+inv', 
        help='"houlsby", "pfeiffer", or "{houlsby/pfeiffer}+inv" config.'
    )
    parser.add_argument(
        '--adapter_non_linearity', 
        default='relu', 
        help='Non-linear activation function for adapters. "relu" or "gelu".'
    )
    parser.add_argument(
        '--adapter_reduction_factor', 
        type=int, 
        default=2, 
        help='Reduction of bottleneck layer. One of [2|16|32].'
    )
    parser.add_argument(
        "--drop_layers",
        nargs='+',
        default=None,
        help="A list of layers for which the adapters should be dropped.",
    )
    parser.add_argument(
        '--cloze_mode', 
        default='maxmin', 
        help='"all" or "maxmin" mode'
    )
    parser.add_argument(
        "--langs",
        nargs='+',
        default=['python', 'java'],
        help="The list of languages considered for the cloze test."
    )
    parser.add_argument(
        '--output_dir', 
        default='../evaluator/predictions/', 
        help='Directory to save output predictions.'
    )
    parser.add_argument(
        '--use_comments', 
        action='store_true', 
        help='If specified, will include code comments with code snippets.'
    )
    parser.add_argument(
        '--train_adapter', 
        action='store_true', 
        help='If specified, will train the adapter setup.'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=None,
        help="A list of language adapters to be fused together.",
    )
    parser.add_argument(
        '--adapter_dir',
        type=str,
        default=None,
        help="Directory to load the language adapters from.",
    )

    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(args.model)
    tokenizer = tokenizer_class.from_pretrained(args.model)
    model = RobertaForMaskedLM.from_pretrained(args.model, from_tf=bool('.ckpt' in args.model), config=config)
    
    model.resize_token_embeddings(len(tokenizer))
    
    global adapter_setup
    adapter_setup = None

    lang_set = args.adapter_name
    # check if adapter already exists, otherwise add it
    if lang_set not in model.config.adapters:
        # resolve the adapter config
        adapter_config = AdapterConfig.load(
            args.adapter_config,
            non_linearity=args.adapter_non_linearity,
            reduction_factor=args.adapter_reduction_factor,
        )
        drop_layers = None
        if args.drop_layers is not None:
            drop_layers = [int(x) for x in args.drop_layers]

        # load pre-trained adapters from local disc or hub
        if args.load_adapter:
            model.load_adapter(
                args.load_adapter,
                config=adapter_config,
                load_as=lang_set,
                leave_out=drop_layers,
            )
        # otherwise, add a fresh adapter
        else:
            raise ValueError(
                "Please specify a suitable model from the hub or"
                "specify the location of pre-trained adapters on your local disc."
                )
    
    model.set_active_adapters([lang_set])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f'cloze test mode: {args.cloze_mode}')
    
    cloze_results = []    
    langs = args.langs or ['ruby', 'javascript', 'go', 'python', 'java', 'php']
    for lang in langs:
      if args.output_dir is not None:
        os.makedirs(os.path.join(args.output_dir, lang), exist_ok=True)
      
      cloze_results.extend(cloze_test(args, lang, model, tokenizer, device))


if __name__ == '__main__':
    main()
