"""Show original posts plus exact model claims that have not been manually reviewed."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--limit', type=int, default=20)
    args.add_argument('--compact', action='store_true')
    ns = args.parse_args()
    inputs = {r['id']: r for r in json.loads((ROOT / 'data/stage2-inputs.json').read_text())['rows']}
    preds = {r['id']: r for r in map(json.loads, (ROOT / 'results/natural/predictions.jsonl').read_text().splitlines())}
    raws = {r['id']: r for r in map(json.loads, (ROOT / 'results/natural/raw.jsonl').read_text().splitlines())}
    notes_file = ROOT / 'review-notes.json'
    notes = json.loads(notes_file.read_text()).get('notes', {}) if notes_file.exists() else {}
    labels = {r['id']: r['label'] for r in json.loads((ROOT / 'frozen/meaning_schema.json').read_text())['core_candidates']}
    todo = [key for key in sorted(preds) if key not in notes][:ns.limit]
    print(json.dumps({'processed': len(preds), 'reviewed': len(notes), 'showing': len(todo)}))
    for ident in todo:
        p = preds[ident]
        print(ident, inputs[ident]['stock_name'], inputs[ident]['published_at'], inputs[ident]['text'])
        if p['status'] == 'ok':
            meanings = p['judgments']; extras = p.get('observations', [])
        else:
            print('OUTPUT_ERROR', p.get('error'))
            try:
                raw = json.loads(raws[ident]['response']['choices'][0]['message']['content'])
                meanings = raw.get('checks', {}); extras = raw.get('observations', [])
            except Exception:
                print('UNREADABLE_RAW'); continue
        for key, value in meanings.items():
            for c in value.get('claims', []):
                if ns.compact:
                    print(labels.get(key, key), c.get('state'), c.get('subject'), c.get('target'), c.get('time'), '证据='+c.get('quote',''), '理由='+c.get('reason','')[:140])
                else:
                    print(labels.get(key, key), json.dumps(c, ensure_ascii=False, separators=(',', ':')))
        for c in extras:
            if ns.compact:
                print('额外',c.get('meaning'),c.get('subject'),c.get('target'),c.get('time'),'证据='+c.get('quote',''),'理由='+c.get('reason','')[:140])
            else:
                print('EXTRA', json.dumps(c, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
