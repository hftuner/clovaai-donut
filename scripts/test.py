import os
import json
import time
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from evaluator import Evaluator

TASK_TYPES = ["classification", "ie", "docvqa"]

def save_json(json_obj, save_path):
    with open(save_path, 'w') as f:
        json.dump(json_obj, f, indent=4)

def run_eval(args):
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model_id = args.pretrained_model_name_or_path
    config = VisionEncoderDecoderConfig.from_pretrained(model_id)
    config.dtype = dtype
    max_length = config.decoder.max_length
    processor = DonutProcessor.from_pretrained(model_id, use_fast=True)
    pretrained_model = VisionEncoderDecoderModel.from_pretrained(model_id, config=config)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()


    # dataset
    test_dataset = load_dataset(args.dataset_name_or_path, split=args.split)
    test_dataset = test_dataset.shuffle(seed=42)
    
    if args.quick_check:
        test_dataset = test_dataset.select(range(args.quick_check_size)) # for debug
    
    # add ground_truth column for classification task
    id2label = None
    if args.task_type == "classification":
        if 'ground_truth' not in test_dataset.column_names:
            id2label = {str(i): c for i, c in enumerate(test_dataset.features['label'].names)}
            def add_ground_truth(example):
                class_name = id2label[str(example['label'])]
                ground_truth = {'gt_parse':{'class': class_name}}
                example['ground_truth'] = json.dumps(ground_truth)
                return example
            test_dataset = test_dataset.map(add_ground_truth)
    
    # store a copy of `gt_parse` in advance
    ground_truths = [ json.loads(gt)['gt_parse'] for gt in test_dataset['ground_truth']]

    # create pixel_values tensors on-the-fly
    def pre_process(examples):
        pixel_values = processor(examples['image'], return_tensors="pt").pixel_values.to(pretrained_model.device)
        return {"pixel_values": pixel_values}
    test_dataset.set_transform(pre_process, columns=["image"])
    
    # dataloader
    batch_size=args.eval_batch_size
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # evaluate
    predictions = []

    for idx, batch in tqdm(enumerate(test_dataloader), desc="Evaluating", total=len(test_dataloader)):
        
        pixel_values = batch['pixel_values']
        bsz = pixel_values.shape[0]

        # decoder_inputs
        prompts = None
        if args.task_type == "classification" or args.task_type == "ie":
            prompts = [args.task_start_token] * bsz
        elif args.task_type == "docvqa":
            prompt_template = args.task_start_token + "<s><s_question>{}</s_question><s_answer>"
            gts = ground_truths[idx: idx + bsz]
            prompts = [prompt_template.format(gt['question']) for gt in gts]
        
        decoder_input_ids = processor.tokenizer(prompts,
                                                 add_special_tokens=False,
                                                 return_tensors="pt",
                                                 padding=True,
                                                 ).input_ids.to(device)
        generated_ids = pretrained_model.generate(pixel_values,
                                        decoder_input_ids=decoder_input_ids,
                                        max_length=max_length,
                                        bad_words_ids=[[processor.tokenizer.unk_token_id]]
                                        )
        token_sequences = processor.batch_decode(generated_ids, skip_special_tokens=False)
        _predictions = [ processor.token2json(seq) for seq in token_sequences ]
        predictions.extend(_predictions)
    
    # evaluation
    evaluator = Evaluator(args.task_type, id2label)
    eval_results = evaluator.eval(predictions, ground_truths)

    # save_results
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    results_save_path  = os.path.join(args.save_dir, 'eval_results.json')
    save_json({
        'results': eval_results,
        'time': timestamp,
        'configs': vars(args)
        },results_save_path
        )
    
    if args.save_predictions:
        predictions_save_path = os.path.join(args.save_dir, 'predictions.json')
        save_json({"predictions": predictions}, predictions_save_path)
        labels_save_path = os.path.join(args.save_dir, 'labels.json')
        save_json({"labels": ground_truths}, labels_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_start_token", type=str, default=None, required=True)
    parser.add_argument("--task_type", type=str, default=None, choices=TASK_TYPES, required=True)
    parser.add_argument("--quick_check", default=False, action="store_true")
    parser.add_argument("--quick_check_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default='./results')
    parser.add_argument("--save_predictions", default=False, action="store_true")
    args, left_argv = parser.parse_known_args()

    run_eval(args)