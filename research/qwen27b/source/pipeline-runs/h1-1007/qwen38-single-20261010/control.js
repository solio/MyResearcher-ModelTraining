const $=id=>document.getElementById(id);
const gb=n=>(Number(n||0)/1e9).toFixed(2);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let csrf='',initialized=false,pending=false,lastState=null;
const formatTime=n=>n?new Date(n*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}):'未知';
function banner(message){$('service-error').textContent=message;$('service-error').classList.toggle('hidden',!message)}
async function session(){
  const r=await fetch('/api/session',{cache:'no-store'});
  if(!r.ok)throw Error('本地控制服务未启动，请运行“启动任务控制台.command”。');
  csrf=(await r.json()).token;
}
async function post(path,body={}){
  if(!csrf)await session();
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify(body)});
  const d=await r.json();if(!r.ok)throw Error(d.error||'请求未完成');return d;
}
function badge(id,text,style=''){const el=$(id);el.textContent=text;el.className='pill '+style;}
function render(d){
  const p=d.phase,live=p==='connected',report=d.report||{},b=report.bf16||{};
  const checked=Number(report.checked_epoch||0),stale=!live||Date.now()/1000-checked>60;
  const bf=report.bf16_trial||{},bs=bf.batches?.startup||{},bn=bf.batches?.natural||{},bc=bf.batches?.controls||{};
  const bfNames={waiting_server:'模型加载／预热',running:bf.batch==='startup'?'启动检查中':'400 条推理中',complete:bf.controls?.state==='running'?'400 条完成，同帖对照中':'本轮已执行完',stopped:'已停止',interrupted:'进程中断'};
  badge('bf400-badge',stale?'上次记录':bfNames[bf.state]||'等待读取',!stale&&bf.state==='complete'?'good':'');
  $('bf400-count').textContent=`新样本已处理 ${bn.attempted||0} / 400 条`;
  $('bf400-bar').value=bn.attempted||0;
  $('bf400-detail').textContent=`${stale?'上次记录：':''}启动检查 ${bs.attempted||0} / 8 条；新样本结构有效 ${bn.valid||0} 条，异常 ${bn.errors||0} 条。${bf.reason?'结束原因：'+bf.reason:''}`+(bf.controls?.state==='running'||bc.attempted?` 旧样本补充对照 ${bc.attempted||0} / 72 条。`:'');
  $('bf400-deadline').textContent=bf.deadline_epoch?'本次推理截止：'+formatTime(bf.deadline_epoch)+'（已预留实例到期前的保存时间）。':'';
  const t=report.trial||{},trialNames={waiting_server:'加载模型',running:'推理中',complete:'本批已处理完',stopped:'已停止',interrupted:'进程中断'};
  badge('trial-badge',stale?'等待实时状态':trialNames[t.state]||'尚未启动',!stale&&t.state==='complete'?'good':!stale&&['stopped','interrupted'].includes(t.state)?'warn':'');
  $('trial-count').textContent=`已处理 ${t.attempted||0} / ${t.expected||80} 条`;
  $('trial-bar').value=t.attempted||0;
  $('trial-detail').textContent=`${stale?'上次记录：':''}结构有效 ${t.valid||0} 条，输出／请求异常 ${t.errors||0} 条。`+(t.mean_seconds?`平均请求耗时 ${t.mean_seconds.toFixed(1)} 秒。`:'')+(t.reason?`结束原因：${t.reason}。`:'');
  $('trial-latest').textContent=t.latest?`最近结果：${t.latest.id} · ${t.latest.status==='ok'?'结构有效':t.latest.error||'输出异常'}。`:'模型就绪后开始写入本批结果。';
  const phaseNames={idle:'未连接',connecting:'正在连接',connected:'已连接',reconnecting:'正在重连',auth_error:'认证失败',remote_error:'任务读取异常',local_error:'本地跟踪异常'};
  badge('connection-badge',phaseNames[p]||p,live?'good':(['reconnecting','auth_error','remote_error','local_error'].includes(p)?'warn':''));
  $('connection-message').textContent=d.message;
  $('endpoint').textContent=d.endpoint?`${d.endpoint.username}@${d.endpoint.host}:${d.endpoint.port}`:'';
  if(!initialized&&d.endpoint){$('host').value=d.endpoint.host;$('port').value=d.endpoint.port;$('username').value=d.endpoint.username;initialized=true;}
  $('resume').disabled=!['connected','remote_error'].includes(p);
  $('disconnect').disabled=['idle','auth_error'].includes(p);
  $('connect').disabled=pending||p==='connecting';
  const names={not_found:'未找到原任务',blocked:'需要检查',starting:'正在恢复',running:'运行中',checking:'正在校验',downloading:'正在下载',paused:'已暂停',complete:'校验通过'};
  const state=b.state||(b.verified?'complete':b.supervisor_alive?'running':'paused');
  badge('task-badge',stale?(checked?'上次记录':'等待连接'):(names[state]||'检查中'),!stale&&state==='complete'?'good':!stale&&['blocked','not_found','paused'].includes(state)?'warn':'');
  const total=Number(b.total_bytes||0),done=Number(b.completed_bytes||0),ratio=total?Math.min(100,100*done/total):0;
  $('bytes').innerHTML=total?`${gb(done)} <small>/ ${gb(total)} GB</small>`:'— <small>/ — GB</small>';
  $('download-bar').value=ratio;$('percent').textContent=total?ratio.toFixed(1)+'%':'—';
  $('task-message').textContent=stale?'这是最后成功读取的记录，不能据此确认远端现在仍在运行。':b.message||'正在检查任务。';
  const speed=Number(b.speed_bytes_per_sec||0);
  $('speed-label').textContent=stale?'上次记录速度':'下载速度';$('speed').textContent=total?(speed/1e6).toFixed(2)+' MB/s':'—';
  if(!stale&&state==='downloading'&&speed>0){
    const mins=Math.ceil((total-done)/speed/60);$('eta').textContent=mins>=60?`${Math.floor(mins/60)} 小时 ${mins%60} 分`:mins+' 分钟';
  }else $('eta').textContent=state==='complete'&&!stale?'已完成':'—';
  $('last-read').textContent='最后成功读取：'+formatTime(checked)+(stale?' · 历史记录':' · 当前连接正常');
  $('verify-note').textContent=b.verified?'已有完整哈希校验记录。'+(stale?'当前连接未核实，请以最后读取时间为准。':'本次连接已检查文件仍在，未重复启动下载。'):'传输字节齐全后还要校验；估计时间不包含最终校验。启动时先核对旧分片，期间速度可能为零。';
  $('hardware').textContent=stale?'等待连接后确认当前硬件。':Array.isArray(report.gpu)&&report.gpu.length?report.gpu.map(g=>`${g.name} · 显存 ${(g.used_mib/1024).toFixed(1)} / ${(g.total_mib/1024).toFixed(0)} GB · 利用率 ${g.utilization}%`).join('；'):'当前未检测到可用 GPU；BF16 下载不需要 GPU。';
  $('disk').textContent=b.free_bytes!=null?(stale?'上次记录：':'')+'磁盘可用空间 '+gb(b.free_bytes)+' GB':'';
  const i=report.inference||{};
  $('remote-inference').textContent=stale?'':i.found===false?'此节点未找到原推理目录；首批结果仍已保存在本地。':`原推理任务：已执行 ${i.attempted_unique||0} / ${i.expected||80} 条，结构有效 ${i.valid_responses||0} 条。${i.complete?'本批已结束。':'此页面仅恢复下载，没有启动推理。'}`;
  $('events').innerHTML=(d.events||[]).map(e=>`<li><time>${esc(new Date(e.time*1000).toLocaleTimeString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}))}</time>${esc(e.message)}</li>`).join('')||'<li class="muted">连接后显示恢复与状态变化。</li>';
}
async function update(){
  try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('无法读取本地控制服务。');lastState=await r.json();render(lastState);banner('');}
  catch(e){if(lastState)render({...lastState,phase:'idle',message:'本地服务连接中断；当前展示上次记录。'});banner('本地控制服务尚未运行或已断开。请运行项目目录中的“启动任务控制台.command”，然后刷新此页。');badge('connection-badge','本地服务未连接','warn');}
}
$('ssh-form').addEventListener('submit',async e=>{
  e.preventDefault();pending=true;$('connect').disabled=true;
  const body={host:$('host').value,port:$('port').value,username:$('username').value,password:$('password').value,resume:$('auto-resume').checked};
  $('password').value='';$('paste-info').value='';
  try{await post('/api/connect',body);$('form-message').textContent='请求已提交，正在检查连接与任务。';$('form-message').className='muted';}
  catch(error){$('form-message').textContent=error.message;$('form-message').className='error';}
  finally{body.password='';pending=false;await update();}
});
$('auto-resume').addEventListener('change',()=>{$('connect').textContent=$('auto-resume').checked?'连接并恢复下载':'连接并查看状态';});
for(const [id,path] of [['resume','/api/resume'],['disconnect','/api/disconnect']])$(id).addEventListener('click',async()=>{
  try{await post(path);$('form-message').textContent='请求已提交。';}catch(e){$('form-message').textContent=e.message;}await update();
});
$('fill-info').addEventListener('click',()=>{
  const s=$('paste-info').value;
  const patterns={host:/(?:SSH地址(?:[（(][^）)]*[）)])?|主机|IP)\s*[：:]\s*([\w.-]+)/i,port:/端口\s*[：:]\s*(\d+)/,username:/(?:SSH账户|SSH账号|用户名|账户)\s*[：:]\s*([^\s]+)/,password:/密码\s*[：:]\s*([^\r\n]+)/};
  let count=0;for(const [id,re] of Object.entries(patterns)){const m=s.match(re);if(m){$(id).value=m[1].trim();count++;}}
  $('paste-info').value='';$('paste-details').open=false;
  $('form-message').textContent=count?'已填入识别到的字段，请连接。':'未识别到连接信息，请直接填写上方字段。';
});
session().then(update).catch(()=>update());setInterval(update,2000);
