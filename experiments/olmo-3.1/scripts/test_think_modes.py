#!/usr/bin/env python
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ollama import chat


def parse_mode(raw: str):
    v = raw.strip().lower()
    if v == 'none':
        return None
    if v == 'false':
        return False
    if v == 'true':
        return True
    return raw


def run_once(model: str, think_value, num_ctx: int, num_predict: int, prompt: str):
    started = time.time()
    error = None
    content = ''
    thinking = None
    try:
        kwargs = dict(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': 'Answer briefly. Return only the final answer text.',
                },
                {'role': 'user', 'content': prompt},
            ],
            options={
                'temperature': 0,
                'num_ctx': num_ctx,
                'num_predict': num_predict,
            },
            stream=False,
        )
        if think_value is not None:
            kwargs['think'] = think_value

        resp = chat(**kwargs)
        content = (resp.message.content or '').strip()
        thinking = getattr(resp.message, 'thinking', None)
    except Exception as e:
        error = f'{type(e).__name__}: {e}'

    return {
        'think': think_value,
        'ok': error is None,
        'error': error,
        'content_preview': content[:200],
        'content_empty': content == '',
        'thinking_is_none': thinking is None,
        'duration_sec': round(time.time() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt20:latest')
    parser.add_argument('--num-ctx', type=int, default=24576)
    parser.add_argument('--num-predict', type=int, default=24)
    parser.add_argument('--prompt', default='What is 2+2?')
    parser.add_argument(
        '--modes',
        nargs='*',
        default=['false', 'true', 'low', 'high'],
        help="think values to test. e.g. false true low high",
    )
    parser.add_argument(
        '--out-dir',
        default='/home/himura_shiro/Projects/BabiLong/experiments/gpt-oss-20b/results',
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tested_modes = [parse_mode(m) for m in args.modes]
    rows = []
    for m in tested_modes:
        print(f'Running think={m!r} ...', flush=True)
        row = run_once(
            model=args.model,
            think_value=m,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            prompt=args.prompt,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        'model': args.model,
        'num_ctx': args.num_ctx,
        'num_predict': args.num_predict,
        'prompt': args.prompt,
        'tested_modes': args.modes,
        'rows': rows,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f'gpt20_think_modes_{ts}.json'
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'Saved: {out_file}')


if __name__ == '__main__':
    main()
