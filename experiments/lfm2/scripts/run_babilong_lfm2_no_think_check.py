#!/usr/bin/env python
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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


def glm_chat_no_think(
    model: str, question: str, context: str, num_predict: int, num_ctx: int
) -> tuple[str, str | None]:
    # GLM 4.7 Flash supports think=false in this environment.
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
        options={
            'temperature': 0,
            'num_predict': num_predict,
            'num_ctx': num_ctx,
        },
        stream=False,
    )
    return (response.message.content or '').strip(), response.message.thinking


def load_existing_records(pred_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not pred_path.exists():
        return records

    with pred_path.open('r', encoding='utf-8') as rf:
        for line_no, raw in enumerate(rf, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f'WARN: skip malformed JSON line {line_no} in {pred_path}')
                continue
            idx = rec.get('index')
            if isinstance(idx, int):
                records[idx] = rec
    return records


def build_summary(
    *,
    args,
    total: int,
    correct_count: int,
    empty_prediction_count: int,
    api_error_count: int,
    thinking_not_none_count: int,
    started: float,
    pred_path: Path,
    run_id: str,
    completed_samples: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    return {
        'dataset': args.dataset,
        'task': args.task,
        'model': args.model,
        'mode': 'no_think',
        'num_ctx': args.num_ctx,
        'num_predict': args.num_predict,
        'num_samples': total,
        'correct_count': correct_count,
        'accuracy': (correct_count / total) if total else 0.0,
        'empty_prediction_count': empty_prediction_count,
        'api_error_count': api_error_count,
        'thinking_not_none_count': thinking_not_none_count,
        'duration_sec': time.time() - started,
        'completed_samples': completed_samples,
        'is_complete': completed_samples >= total,
        'checkpoint_every': checkpoint_every,
        'run_id': run_id,
        'predictions_file': str(pred_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='/home/himura_shiro/Projects/BabiLong/data/babilong-1k-samples/1k/qa1-00000-of-00001.parquet',
    )
    parser.add_argument('--task', default='qa1')
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--model', default='lfm2:latest')
    parser.add_argument('--num-predict', type=int, default=24)
    parser.add_argument('--num-ctx', type=int, default=24576)
    parser.add_argument('--out-dir', default='/home/himura_shiro/Projects/BabiLong/experiments/lfm2/results')
    parser.add_argument(
        '--run-id',
        default=None,
        help='Stable id for resumable run. Default: current timestamp for a new run.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing predictions/state files for the same run-id.',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=8,
        help='Write state/summary checkpoint every N processed samples.',
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Retry count for transient Ollama runner failures per sample.',
    )
    parser.add_argument(
        '--retry-sleep-sec',
        type=float,
        default=2.0,
        help='Sleep seconds between retries.',
    )
    args = parser.parse_args()

    if args.task not in TASK_LABELS:
        raise ValueError(f'Unknown task: {args.task}')
    if args.checkpoint_every <= 0:
        raise ValueError('--checkpoint-every must be > 0')
    if args.max_retries < 0:
        raise ValueError('--max-retries must be >= 0')
    if args.retry_sleep_sec < 0:
        raise ValueError('--retry-sleep-sec must be >= 0')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_first_n_samples(Path(args.dataset), args.num_samples)
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = f'babilong_{args.task}_n{len(samples)}_{args.model}_no_think'
    pred_path = out_dir / f'{run_tag}_predictions_{run_id}.jsonl'
    summary_path = out_dir / f'{run_tag}_summary_{run_id}.json'
    state_path = out_dir / f'{run_tag}_state_{run_id}.json'

    if args.resume and not pred_path.exists():
        raise FileNotFoundError(
            f'--resume was set but predictions file is missing: {pred_path}. '
            'Use the same --run-id as previous run.'
        )
    if not args.resume and pred_path.exists():
        raise FileExistsError(
            f'Predictions file already exists: {pred_path}. '
            'Use --resume to continue or pass a new --run-id.'
        )

    existing_records = load_existing_records(pred_path) if args.resume else {}
    completed_indices = set(existing_records.keys())

    correct_count = sum(1 for r in existing_records.values() if r.get('correct') is True)
    empty_prediction_count = sum(1 for r in existing_records.values() if r.get('prediction_is_empty') is True)
    api_error_count = sum(1 for r in existing_records.values() if r.get('api_error'))
    thinking_not_none_count = sum(1 for r in existing_records.values() if r.get('thinking') is not None)
    started = time.time()
    resumed_count = len(completed_indices)

    print(
        f'Run id={run_id} resume={args.resume} resumed_samples={resumed_count}/{len(samples)} '
        f'checkpoint_every={args.checkpoint_every} max_retries={args.max_retries}',
        flush=True,
    )

    def write_checkpoint() -> None:
        completed_samples = len(completed_indices)
        summary = build_summary(
            args=args,
            total=len(samples),
            correct_count=correct_count,
            empty_prediction_count=empty_prediction_count,
            api_error_count=api_error_count,
            thinking_not_none_count=thinking_not_none_count,
            started=started,
            pred_path=pred_path,
            run_id=run_id,
            completed_samples=completed_samples,
            checkpoint_every=args.checkpoint_every,
        )
        state = {
            'run_id': run_id,
            'run_tag': run_tag,
            'last_completed_index': max(completed_indices) if completed_indices else 0,
            'completed_samples': completed_samples,
            'num_samples': len(samples),
            'checkpoint_every': args.checkpoint_every,
            'summary_file': str(summary_path),
            'predictions_file': str(pred_path),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        with state_path.open('w', encoding='utf-8') as sf:
            json.dump(state, sf, ensure_ascii=False, indent=2)
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    with pred_path.open('a', encoding='utf-8') as wf:
        processed_since_checkpoint = 0
        for s in samples:
            if s['index'] in completed_indices:
                continue

            error_msg = None
            pred = ''
            thinking = None
            attempts = 0
            for attempt in range(args.max_retries + 1):
                attempts = attempt + 1
                try:
                    pred, thinking = glm_chat_no_think(
                        args.model, s['question'], s['context'], args.num_predict, args.num_ctx
                    )
                    error_msg = None
                    break
                except Exception as e:
                    error_msg = f'{type(e).__name__}: {e}'
                    retryable = 'model runner has unexpectedly stopped' in error_msg.lower()
                    has_retry = attempt < args.max_retries
                    if retryable and has_retry:
                        print(
                            f"[{s['index']}/{len(samples)}] retry {attempt + 1}/{args.max_retries} "
                            f"after runner-stop error",
                            flush=True,
                        )
                        if args.retry_sleep_sec > 0:
                            time.sleep(args.retry_sleep_sec)
                        continue
                    break

            if error_msg is not None:
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
                'attempts': attempts,
            }
            wf.write(json.dumps(record, ensure_ascii=False) + '\n')
            wf.flush()
            completed_indices.add(s['index'])
            processed_since_checkpoint += 1
            print(
                f"[{s['index']}/{len(samples)}] correct={bool(correct)} empty={pred == ''} "
                f"thinking={thinking is not None} attempts={attempts} pred={pred[:90]}",
                flush=True,
            )
            if processed_since_checkpoint >= args.checkpoint_every:
                write_checkpoint()
                processed_since_checkpoint = 0

        if processed_since_checkpoint > 0:
            write_checkpoint()

    total = len(samples)
    summary = build_summary(
        args=args,
        total=total,
        correct_count=correct_count,
        empty_prediction_count=empty_prediction_count,
        api_error_count=api_error_count,
        thinking_not_none_count=thinking_not_none_count,
        started=started,
        pred_path=pred_path,
        run_id=run_id,
        completed_samples=len(completed_indices),
        checkpoint_every=args.checkpoint_every,
    )
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n=== Summary ===')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved predictions: {pred_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved state: {state_path}')


if __name__ == '__main__':
    main()
