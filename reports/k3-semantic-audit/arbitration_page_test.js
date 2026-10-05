const { JSDOM } = require('jsdom');
const fs = require('fs');

const html = fs.readFileSync('/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/contrarian-bottleneck-20261005/arbitration.html', 'utf8');

// 提供 localStorage(jsdom 默认有,但确保干净)
const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/' });
const { window } = dom;
const doc = window.document;

let pass = 0, fail = 0;
const t = (name, cond, extra='') => { cond ? (pass++, console.log('  PASS', name)) : (fail++, console.log('  FAIL', name, extra)); };

// 1. 卡片数
const cards = doc.querySelectorAll('.card');
console.log('1) 初始渲染');
t('渲染 87 张卡片', cards.length === 87, '实际=' + cards.length);
t('硬分歧卡片有 hard 类', doc.querySelectorAll('.card.hard').length === 16, '实际=' + doc.querySelectorAll('.card.hard').length);
t('首张卡片是硬分歧', cards[0].classList.contains('hard'));
t('进度 0/87', doc.getElementById('progTxt').textContent.trim() === '0 / 87', '实际=' + doc.getElementById('progTxt').textContent);

// 2. 点击判定按钮
console.log('2) 点击「K3 对」按钮');
const firstCard = cards[0];
const sid = firstCard.dataset.id;
const k3btn = firstCard.querySelector('.vbtn.k3');
k3btn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
const stored = JSON.parse(window.localStorage.getItem('k3-arbitration-v1') || '{}');
t('localStorage 写入 verdict=k3', stored[sid] && stored[sid].verdict === 'k3', JSON.stringify(stored[sid]));
const newFirst = doc.querySelector(`.card[data-id="${sid}"]`);
t('卡片标记为 done', newFirst.classList.contains('done'));
t('显示已判文案', /已判/.test(newFirst.textContent) && /K3 对/.test(newFirst.textContent));
t('进度更新为 1/87', doc.getElementById('progTxt').textContent.trim() === '1 / 87', '实际=' + doc.getElementById('progTxt').textContent);
t('底栏统计更新', /已判 <b>1<\/b>/.test(doc.getElementById('footStat').innerHTML));

// 3. 筛选
console.log('3) 筛选：方向硬分歧');
doc.querySelector('.chip[data-f="hard"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
t('筛出 16 条', doc.querySelectorAll('.card').length === 16, '实际=' + doc.querySelectorAll('.card').length);

console.log('4) 筛选：未判');
doc.querySelector('.chip[data-f="undone"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
t('未判 86 条', doc.querySelectorAll('.card').length === 86, '实际=' + doc.querySelectorAll('.card').length);

console.log('5) 筛选：K3判反讽');
doc.querySelector('.chip[data-f="sar"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
t('反讽 12 条(包内)', doc.querySelectorAll('.card').length === 12, '实际=' + doc.querySelectorAll('.card').length);

// 6. 键盘仲裁
console.log('6) 键盘 1/2 判定');
doc.querySelector('.chip[data-f="all"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
const rows = [...doc.querySelectorAll('.card')];
const sid2 = rows[0].dataset.id;
window.document.dispatchEvent(new window.KeyboardEvent('keydown', { key: '2', bubbles: true }));
const st2 = JSON.parse(window.localStorage.getItem('k3-arbitration-v1') || '{}');
t('键盘 2 = 学生对，写入成功', st2[sid2] && st2[sid2].verdict === 'student', JSON.stringify(st2[sid2]));

// 7. 备注输入
console.log('7) 备注输入');
const inp = doc.querySelector(`.card[data-id="${sid2}"] .note input`);
if (inp) {
  inp.value = '测试备注';
  inp.dispatchEvent(new window.Event('input', { bubbles: true }));
  const st3 = JSON.parse(window.localStorage.getItem('k3-arbitration-v1'));
  t('备注写入', st3[sid2] && st3[sid2].note === '测试备注', JSON.stringify(st3[sid2]));
} else { t('备注输入框存在', false); }

console.log(`\n结果: ${pass} 通过, ${fail} 失败`);
process.exit(fail ? 1 : 0);
