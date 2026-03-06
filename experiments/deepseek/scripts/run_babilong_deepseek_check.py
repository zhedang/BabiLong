#!/usr/bin/env python
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request

import pyarrow.parquet as pq

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


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('#') or '=' not in s:
            continue
        k, v = s.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


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
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    samples = []
    for idx, row in enumerate(rows[:n], start=1):
        samples.append(
            {
                'index': idx,
                'question': (row.get('question') or '').strip(),
                'context': (row.get('input') or '').strip(),
                'target': (row.get('target') or '').strip(),
            }
        )
    return samples


def deepseek_chat(base_url: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        'model': model,
        'temperature': 0,
        'messages': [
            {
                'role': 'system',
                'content': 'Answer using only the given context. Output only the final answer text.',
            },
            {'role': 'user', 'content': prompt},
        ],
    }
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )

    with request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode('utf-8'))

    choices = body.get('choices') or []
    if not choices:
        return ''
    message = (choices[0] or {}).get('message') or {}
    return (message.get('content') or '').strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='/home/himura_shiro/Projects/BabiLong/data/babilong-1k-samples/1k/qa1-00000-of-00001.parquet',
    )
    parser.add_argument('--task', default='qa1')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--model', default='deepseek-chat')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--out-dir', default='/home/himura_shiro/Projects/BabiLong/experiments/deepseek/results')
    args = parser.parse_args()

    if args.task not in TASK_LABELS:
        raise ValueError(f'Unknown task: {args.task}')

    script_dir = Path(__file__).resolve().parent
    load_env(script_dir / '.env')
    load_env(Path('/home/himura_shiro/Projects/SQuAD 2.0/experiments/DeepSeek-V3.2/.env'))

    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    if not api_key:
        raise RuntimeError('DEEPSEEK_API_KEY is missing in environment or .env')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_first_n_samples(Path(args.dataset), args.num_samples)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = f'babilong_{args.task}_n{len(samples)}_{args.model}'
    pred_path = out_dir / f'{run_tag}_predictions_{timestamp}.jsonl'
    summary_path = out_dir / f'{run_tag}_summary_{timestamp}.json'

    correct_count = 0
    empty_prediction_count = 0
    api_error_count = 0
    started = time.time()

    with pred_path.open('w', encoding='utf-8') as wf:
        for s in samples:
            prompt = f"Context:\n{s['context']}\n\nQuestion:\n{s['question']}"
            error_msg = None
            try:
                pred = deepseek_chat(base_url, api_key, args.model, prompt, args.timeout)
            except error.HTTPError as e:
                detail = e.read().decode('utf-8', errors='replace')
                pred = ''
                error_msg = f'HTTP {e.code}: {detail[:300]}'
                api_error_count += 1
            except Exception as e:
                pred = ''
                error_msg = f'{type(e).__name__}: {e}'
                api_error_count += 1

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
                'api_error': error_msg,
                'correct': bool(correct),
            }
            wf.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(
                f"[{s['index']}/{len(samples)}] correct={bool(correct)} "
                f"empty={pred == ''} pred={pred[:90]}"
            )

    total = len(samples)
    summary = {
        'dataset': args.dataset,
        'task': args.task,
        'model': args.model,
        'num_samples': total,
        'correct_count': correct_count,
        'accuracy': (correct_count / total) if total else 0.0,
        'empty_prediction_count': empty_prediction_count,
        'api_error_count': api_error_count,
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
