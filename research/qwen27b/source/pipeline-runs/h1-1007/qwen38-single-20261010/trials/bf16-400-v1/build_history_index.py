from pathlib import Path
import gzip,hashlib,json,sqlite3,time,os
r=Path('/root/qwen38-single-20261010/trials/bf16-400-v1');m=json.loads((r/'manifest.json').read_text());z=r/'source-snapshot-20261009.json.gz'
assert hashlib.file_digest(z.open('rb'),'sha256').hexdigest()==m['gzip_sha256']
with gzip.open(z,'rb') as f: raw=f.read()
assert hashlib.sha256(raw).hexdigest()==m['source_sha256']
rows=json.loads(raw);del raw
p=r/'history-index.sqlite.tmp';db=sqlite3.connect(p)
db.execute('CREATE TABLE posts (stock_code TEXT, source_item_id TEXT, published_at TEXT, text TEXT, text_hash TEXT, url TEXT, source TEXT)')
db.executemany('INSERT INTO posts VALUES (?,?,?,?,?,?,?)',((x.get('stock_code'),x.get('source_item_id'),x.get('published_at'),x.get('text'),x.get('text_hash'),x.get('url'),x.get('source')) for x in rows))
db.execute('CREATE INDEX stock_time ON posts(stock_code,published_at)');db.execute('CREATE INDEX text_identity ON posts(text_hash)')
db.commit();check=db.execute('PRAGMA quick_check').fetchone()[0];count=db.execute('SELECT count(*) FROM posts').fetchone()[0];db.close()
assert check=='ok' and count==m['records'];p.replace(r/'history-index.sqlite')
(r/'history-index-verification.json').write_text(json.dumps({'rows':count,'sqlite_check':check,'source_gzip_sha256':m['gzip_sha256'],'source_json_sha256':m['source_sha256'],'finished_at_epoch':time.time(),'usage':'Historical stock/date retrieval and identity exclusion only; adjacent stock posts are not assumed to be conversational context. No inference performed.'},indent=2))
print('verified and indexed',count,flush=True)
