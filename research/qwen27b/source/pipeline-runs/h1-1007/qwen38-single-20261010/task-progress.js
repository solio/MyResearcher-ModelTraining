(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const number = n => Number(n).toLocaleString('zh-CN');
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const time = epoch => Number.isFinite(epoch) ? new Date(epoch * 1000).toLocaleString('zh-CN', {timeZone:'Asia/Shanghai', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false}) : '尚无记录';
  const lastGood = new Map();
  let busy = false;
  async function read(path) {
    const response = await fetch(path, {cache:'no-store', signal:AbortSignal.timeout(8000)});
    if (!response.ok) throw Error('读取失败');
    return response.json();
  }
  async function loadTask(task) {
    const [backup, progress] = await Promise.allSettled([read(task.backup_path), read(task.progress_path)]);
    const previous = lastGood.get(task.id) || {};
    const snapshot = {
      backup: backup.status === 'fulfilled' ? backup.value : previous.backup,
      progress: progress.status === 'fulfilled' ? progress.value : previous.progress,
      readFailed: backup.status !== 'fulfilled'
    };
    if (backup.status === 'fulfilled') lastGood.set(task.id, snapshot);
    const b = snapshot.backup;
    if (!b) return {...task, unknown:true, label:'等待读取', tone:'warn', readFailed:true};
    const received = b.complement_received ?? b.rows;
    const freshCount = b.new_rows ?? b.rows;
    if (!Number.isFinite(received) || received < 0 || received > task.expected || !Number.isFinite(freshCount) || freshCount < 0 || freshCount > task.new_expected) {
      return {...task, unknown:true, label:'计数待核实', tone:'warn', readFailed:true};
    }
    const verified = b.all_download_hashes_verified === true;
    const complete = b.state === 'complete' && received === task.expected && verified;
    const stale = !complete && (snapshot.readFailed || !b.synced_epoch || Date.now()/1000 - b.synced_epoch > 90);
    const states = {running:'正在运行', queued_after_top5:'排队等待', complete:'完成状态待核实', stopped:'已停止', interrupted:'运行中断', loading_model:'加载模型'};
    const label = complete ? '完成并已保存' : stale ? '状态待核实' : states[b.state] || '状态待核实';
    return {...task, ...snapshot, received, freshCount, reused:b.reused_prior ?? 0, verified, complete, stale, label, tone:complete || (!stale && b.state === 'running') ? 'good' : 'warn'};
  }
  function renderHero(task, allDone, tasks) {
    el('current-caption').textContent = allDone ? '本轮已全部完成' : task.unknown || task.stale ? '当前任务 · 状态待核实' : task.backup.state === 'running' ? '当前正在执行' : '当前待完成任务';
    el('current-title').textContent = allDone ? '两批回答均已保存到本地' : task.title;
    el('current-description').textContent = allDone ? '各批次的完整回答可在下方任务列表下载。' : task.description;
    el('current-state').textContent = allDone ? '全部完成' : task.label;
    el('current-state').className = 'pill ' + task.tone;
    if (task.unknown) {
      el('current-count').textContent = '—';
      el('current-breakdown').textContent = '暂未读取到备份记录，不将缺失数据显示为 0 条。';
      el('current-time').textContent = '尚无可核实的时间';
      el('current-proof').textContent = '';
      el('current-bar').value = 0;
      el('current-download').classList.add('hidden');
      el('current-warning').textContent = '读取失败，页面会自动重试。';
      el('current-warning').classList.remove('hidden');
      return;
    }
    const count = allDone ? tasks.reduce((sum, t) => sum + t.received, 0) : task.received;
    const total = allDone ? tasks.reduce((sum, t) => sum + t.expected, 0) : task.expected;
    el('current-count').innerHTML = `${number(count)} <small>/ ${number(total)} 条</small>`;
    el('current-bar').max = total;
    el('current-bar').value = count;
    el('current-breakdown').textContent = allDone ? '包含 6 条此前相同输入的回执；逐条复用出处保留在结果内。' : task.reused ? `${number(task.freshCount)} / ${number(task.new_expected)} 条新推理 + ${number(task.reused)} 条旧回执；还缺 ${number(total-count)} 条。` : `还缺 ${number(total-count)} 条；昨天和今天的回答合并计数。`;
    el('current-time').textContent = time(task.backup.synced_epoch);
    el('current-proof').textContent = task.complete ? '完整备份已校验；完成后无需持续更新。' : task.stale ? '这是最后成功备份的时间，不能确认此刻仍在运行。' : task.verified ? '本地备份哈希校验通过，后台持续下载中。' : '备份哈希尚未确认。';
    el('current-download').href = task.result_path;
    el('current-download').textContent = task.complete ? '下载本批完整回答 JSONL' : '下载本批已保存回答 JSONL';
    el('current-download').classList.toggle('hidden', allDone || task.received === 0);
    const warning = task.stale ? '备份超过 90 秒未更新或读取失败。保留最后已保存数量，当前远端运行状态待核实。' : task.readFailed ? '当前读取失败，展示此前已核实的完整备份记录。' : !task.verified ? '备份校验状态待确认。' : '';
    el('current-warning').textContent = warning;
    el('current-warning').classList.toggle('hidden', !warning);
  }
  function renderRows(tasks, current) {
    el('queue-rows').innerHTML = tasks.map((task, i) => {
      const count = task.unknown ? '—' : `${number(task.received)} / ${number(task.expected)}`;
      const note = task.unknown ? '读取中' : task.reused ? `${number(task.freshCount)} 条新推理 + ${task.reused} 条复用` : task.complete ? '完整原始回答已保存' : '按已下载回答计数';
      const detail = task.unknown ? '' : `<small>保存于 ${escape(time(task.backup.synced_epoch))}</small>`;
      return `<tr${current.id === task.id && !task.complete ? ' class="current-row"' : ''}><td><strong>${i+1}. ${escape(task.title)}</strong><small>${escape(note)}</small></td><td><span class="pill ${escape(task.tone)}">${escape(task.label)}</span>${detail}</td><td><strong>${count}</strong><progress aria-label="${escape(task.title)}已保存进度" max="${task.expected}" value="${task.received || 0}"></progress></td><td>${!task.unknown && task.received > 0 ? `<a download href="${escape(task.result_path)}">${task.complete ? '完整结果' : '当前结果'} JSONL ↓</a>` : '暂无结果'}</td></tr>`;
    }).join('');
  }
  async function refresh() {
    if (busy) return;
    busy = true;
    el('queue-refresh').disabled = true;
    try {
      const config = await read('task-queue.json');
      const tasks = await Promise.all(config.tasks.map(loadTask));
      if (!tasks.length) throw Error('没有配置任务');
      const active = tasks.find(t => !t.complete && t.backup?.state === 'running') || tasks.find(t => !t.complete) || tasks[tasks.length-1];
      const allDone = tasks.every(t => t.complete);
      el('queue-session').textContent = config.session_label;
      el('queue-scope').textContent = config.scope;
      renderHero(active, allDone, tasks);
      renderRows(tasks, active);
      el('queue-total').textContent = tasks.every(t => !t.unknown) ? `本轮合计已保存 ${number(tasks.reduce((sum,t) => sum+t.received,0))} / ${number(tasks.reduce((sum,t) => sum+t.expected,0))} 条` : '部分任务状态尚未读取';
      el('queue-history-links').innerHTML = (config.history || []).map(item => `<div><a href="${escape(item.href)}">${escape(item.title)}</a><small>${escape(item.kind)}</small></div>`).join('');
      el('queue-read-time').textContent = `页面读取于 ${time(Date.now()/1000)} · 自动刷新每 10 秒；后台备份约每 20 秒。`;
      el('queue-global-error').classList.add('hidden');
      try {
        const budget = await read(config.budget_path);
        el('queue-budget').textContent = `本轮时限：${time(budget.no_new_requests_after_epoch)} 停止派发新请求，预留收尾和备份时间。`;
      } catch { el('queue-budget').textContent = '暂未读取到本轮时限。'; }
    } catch {
      el('queue-global-error').textContent = '暂时无法刷新任务列表；已有数字是上次读取的记录，不能据此确认当前运行。页面会继续重试。';
      el('queue-global-error').classList.remove('hidden');
      el('current-state').textContent = '刷新失败';
      el('current-state').className = 'pill warn';
    } finally {
      busy = false;
      el('queue-refresh').disabled = false;
    }
  }
  el('queue-refresh').addEventListener('click', refresh);
  let toolsLoaded = false;
  el('legacy-tools').addEventListener('toggle', () => {
    if (!el('legacy-tools').open || toolsLoaded) return;
    toolsLoaded = true;
    const script = document.createElement('script');
    script.src = 'control.js';
    document.body.appendChild(script);
  });
  refresh();
  setInterval(refresh, 10000);
})();
