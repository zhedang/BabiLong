#!/usr/bin/env python
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from ollama import chat

# Official labels from BABILong metrics.py
TASK_LABELS = {
    'qa1': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa2': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa3': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa4': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa5': ['Bill', 'Fred', 'Jeff', 'Mary', 'apple', 'football', 'milk'],
    'qa6': ['no', 'yes'],
    'qa7': ['none', 'one', 'three', 'two'],
    'qa8': ['apple', 'football', 'milk', 'nothing'],
    'qa9': ['no', 'yes'],
    'qa10': ['maybe', 'no', 'yes'],
    'qa11': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa12': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa13': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa14': ['bedroom', 'cinema', 'kitchen', 'office', 'park', 'school'],
    'qa15': ['cat', 'mouse', 'sheep', 'wolf'],
    'qa16': ['gray', 'green', 'white', 'yellow'],
    'qa17': ['no', 'yes'],
    'qa18': ['no', 'yes'],
    'qa19': ['e,e', 'e,n', 'e,s', 'n,e', 'n,n', 'n,w', 's,e', 's,s', 's,w', 'w,n', 'w,s', 'w,w'],
    'qa20': ['bedroom', 'bored', 'garden', 'hungry', 'kitchen', 'thirsty', 'tired'],
}


def preprocess_output(output: str) -> str:
    output = output.lower()
    output = output.split('.')[0]
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('Question')[0]
    return output


def compare_answers(target: str, output: str, question: str, task_labels) -> bool:
    output = preprocess_output(output)
    target = (target or '').lower()
    labels = {label.lower() for label in task_labels}

    labels_in_question = {label for label in labels if label in (question or '').lower()}
    labels_in_output = {label for label in labels if label in output}
    labels_in_output = labels_in_output - labels_in_question

    if ',' in target and len(target) > 3:
        subtargets = target.split(',')
        num_subtargets = len(subtargets)
        if all(t in labels_in_output for t in subtargets) and len(labels_in_output) == num_subtargets:
            return True
    else:
        if target in labels_in_output and len(labels_in_output) == 1:
            return True

    return False


def load_first_n_samples(parquet_path: Path, n: int):
    pf = pq.ParquetFile(parquet_path)
    samples = []
    idx = 1
    for batch in pf.iter_batches(batch_size=min(64, max(1, n))):
        for row in batch.to_pylist():
            samples.append(
                {
                    'index': idx,
                    'question': (row.get('question') or '').strip(),
                    'context': (row.get('input') or '').strip(),
                    'target': (row.get('target') or '').strip(),
                }
            )
            idx += 1
            if len(samples) >= n:
                return samples
    return samples


def qwen_chat_no_think(model: str, question: str, context: str, num_ctx: int) -> tuple[str, str | None]:
    response = chat(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': 'Answer using only the given context. Output only the final answer text.',
            },
            {
                'role': 'user',
                'content': f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        think=False,
        options={'num_ctx': num_ctx},
        stream=False,
    )
    return (response.message.content or '').strip(), response.message.thinking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='/home/himura_shiro/Projects/BabiLong/data/babilong-1k-samples/16k/qa1-00000-of-00001.parquet',
    )
    parser.add_argument('--task', default='qa1')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--model', default='qwen3.5:4b')
    parser.add_argument('--num-ctx', type=int, default=24576)
    parser.add_argument('--out-dir', default='/home/himura_shiro/Projects/BabiLong/experiments/qwen3.5-4b/results')
    args = parser.parse_args()

    if args.task not in TASK_LABELS:
        raise ValueError(f'Unknown task: {args.task}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_first_n_samples(Path(args.dataset), args.num_samples)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = f'babilong_{args.task}_n{len(samples)}_{args.model}_no_think_ctx{args.num_ctx}'
    pred_path = out_dir / f'{run_tag}_predictions_{timestamp}.jsonl'
    summary_path = out_dir / f'{run_tag}_summary_{timestamp}.json'

    correct_count = 0
    empty_prediction_count = 0
    api_error_count = 0
    thinking_not_none_count = 0
    started = time.time()

    with pred_path.open('w', encoding='utf-8') as wf:
        for s in samples:
            error_msg = None
            pred = ''
            thinking = None
            try:
                pred, thinking = qwen_chat_no_think(args.model, s['question'], s['context'], args.num_ctx)
            except Exception as e:
                error_msg = f'{type(e).__name__}: {e}'
                api_error_count += 1

            if thinking is not None:
                thinking_not_none_count += 1
            if pred == '':
                empty_prediction_count += 1

            correct = compare_answers(
                target=s['target'],
                output=pred,
                question=s['question'],
                task_labels=TASK_LABELS[args.task],
            )
            correct_count += int(correct)

            record = {
                'index': s['index'],
                'question': s['question'],
                'target': s['target'],
                'prediction': pred,
                'prediction_is_empty': pred == '',
                'thinking': thinking,
                'api_error': error_msg,
                'correct': bool(correct),
            }
            wf.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(
                f"[{s['index']}/{len(samples)}] correct={bool(correct)} empty={pred == ''} "
                f"thinking={thinking is not None} pred={pred[:90]}"
            )

    total = len(samples)
    summary = {
        'dataset': args.dataset,
        'task': args.task,
        'model': args.model,
        'mode': 'no_think',
        'num_ctx': args.num_ctx,
        'num_samples': total,
        'correct_count': correct_count,
        'accuracy': (correct_count / total) if total else 0.0,
        'empty_prediction_count': empty_prediction_count,
        'api_error_count': api_error_count,
        'thinking_not_none_count': thinking_not_none_count,
        'duration_sec': time.time() - started,
        'predictions_file': str(pred_path),
    }

    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n=== Summary ===')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved predictions: {pred_path}')
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    main()
