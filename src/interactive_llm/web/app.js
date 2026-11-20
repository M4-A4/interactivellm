/* Realtime voice/vision client.
 * Protocol: binary frames out = 0x01 + int16 16 kHz mic PCM, 0x02 + JPEG camera frame; binary in = 0x03 + int16 24 kHz TTS PCM.
 * JSON messages carry state, transcripts, latency and control (see onMessage and send). */

/* ---------- Shared state and small helpers ---------- */
/** Shorthand for document.getElementById. */
const $ = (id) => document.getElementById(id);
// ws: WebSocket; micCtx/playCtx: capture and playback AudioContexts; micNode/playNode: their worklet nodes (worklets.js);
// stream: camera+mic MediaStream; frameTimer: interval that uploads camera frames.
let ws, micCtx, playCtx, micNode, playNode, stream, frameTimer;
// ending: user chose to end, so the socket close is expected; micOn/camOn/speakOn mirror the toggle buttons;
// currentBotBubble: the streaming assistant message; the *Timer vars auto-hide transient badges.
let ending = false, speakerBadgeTimer, micOn = true, camOn = true, speakOn = true, currentBotBubble = null, framesSentTimer;

// Offscreen canvas used to downscale camera frames before upload.
const canvas = document.createElement('canvas');
const frameContext = canvas.getContext('2d');

/** Show the assistant state (idle | listening | thinking | speaking) in the pill and on the video frame. */
function setPill(state) {
  $('pill').className = 'pill ' + state; $('pillText').textContent = state;
  const videoWrap = document.querySelector('.video-wrap'); if (videoWrap) videoWrap.dataset.state = state;
}
/** Set the connection dot's CSS class ('', 'connecting', 'live' or 'error'). */
function setConn(cls) { $('conn').className = 'dot ' + cls; }
// Auto-scroll only while the user is already near the bottom, so reading history is not interrupted.
function nearBottom() { const log = $('log'); return log.scrollHeight - log.scrollTop - log.clientHeight < 80; }
/** Scroll the chat log to the bottom if `force` or the user is already near it. */
function scrollLog(force) { if (force || nearBottom()) $('log').scrollTop = 1e9; }
/** Append a chat message ('user' | 'bot' | 'err') and return its bubble element (text is set via textContent only). */
function addMsg(cls, text) {
  const log = $('log'), stick = nearBottom();
  $('empty').hidden = true;
  const row = document.createElement('div'); row.className = 'msg ' + cls;
  if (cls !== 'err') {
    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = (cls === 'user' ? 'You' : 'Assistant') + ' · ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    row.appendChild(meta);
  }
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
  row.appendChild(bubble); log.appendChild(row);
  if (stick || cls === 'user') log.scrollTop = 1e9;
  return bubble;
}
/** Enable or disable the conversation controls. */
function setEnabled(on) {
  for (const id of ['mic', 'cam', 'speak', 'stop', 'backend', 'text', 'send']) $(id).disabled = !on;
}

/* ---------- Media setup and connection lifecycle ---------- */
// Camera settings from /api/settings: capture resolution ("WxH") and the longest side of frames sent to the model.
const cameraSettings = { resolution: '640x480', maxSide: 448 };
/** Copy the camera-related values from a /api/settings payload into `cameraSettings`. */
function adoptCameraSettings(settings) {
  cameraSettings.resolution = settings.camera_resolution; cameraSettings.maxSide = settings.frame_max_side;
}

// ---- Camera hardware discovery ----
// Common capture sizes; the dropdown offers those the camera can actually deliver.
const STANDARD_RESOLUTIONS = [[320, 240, '240p'], [640, 480, '480p'], [1280, 720, '720p'], [1920, 1080, '1080p'], [2560, 1440, '1440p'], [3840, 2160, '4K']];
// Largest size the camera reports: { maxWidth, maxHeight, label } or null if unknown. Cached so the list is ready before permission.
let cameraCapabilities = null;
try { cameraCapabilities = JSON.parse(localStorage.getItem('cameraCapabilities')); } catch (e) { /* storage unavailable */ }

/** Record what a live video track reports it can do, remember it, and refresh the settings dropdown. */
function learnCameraCapabilities(track) {
  const limits = track.getCapabilities ? track.getCapabilities() : {};
  if (!limits.width || !limits.height) return;
  cameraCapabilities = { maxWidth: limits.width.max, maxHeight: limits.height.max, label: track.label };
  try { localStorage.setItem('cameraCapabilities', JSON.stringify(cameraCapabilities)); } catch (e) { /* ignore */ }
  refreshCameraOptions();
}
/** At page load (and when a camera is plugged in or out), briefly open the camera to read its limits, if already permitted. */
async function probeCamera() {
  try {
    if (navigator.permissions && (await navigator.permissions.query({ name: 'camera' })).state !== 'granted') return;  // never prompt at load
    if (stream) return;  // a conversation already holds the camera and has reported its limits
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    learnCameraCapabilities(probe.getVideoTracks()[0]);
    probe.getTracks().forEach((track) => track.stop());
  } catch (e) { /* camera busy or absent: keep cached or default options */ }
}
/** Resolution choices [value, label] supported by the detected camera (the common set when nothing was detected). */
function cameraResolutionChoices() {
  const known = cameraCapabilities;
  const supported = STANDARD_RESOLUTIONS.filter(([width, height]) => !known || (width <= known.maxWidth && height <= known.maxHeight));
  return (supported.length ? supported : STANDARD_RESOLUTIONS.slice(0, 3)).map(([width, height, name]) => [`${width}x${height}`, `${name} (${width} × ${height})`]);
}
/** Rebuild the camera dropdown and its note from the detected hardware, keeping the current choice if still valid. */
function refreshCameraOptions() {
  const select = $('cfgCamRes'); if (!select) return;
  const choices = cameraResolutionChoices();
  const wanted = select.value || cameraSettings.resolution;
  const valid = choices.some(([value]) => value === wanted);
  fillSelect(select, choices, valid ? wanted : choices[choices.length > 1 ? 1 : 0][0]);
  $('cfgCamNote').textContent = cameraCapabilities
    ? `${cameraCapabilities.label || 'camera'}: up to ${choices[choices.length - 1][1]}`
    : 'camera not detected yet; allow camera access, then reopen settings';
}
/** getUserMedia video constraints for the chosen resolution. */
function videoConstraints() {
  const [width, height] = cameraSettings.resolution.split('x').map(Number);
  return { width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: 15 } };
}

/** Get mic (+ camera if allowed); falls back to audio-only when the camera is unavailable. */
async function getMedia() {
  const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
  try {
    return await navigator.mediaDevices.getUserMedia({ audio, video: videoConstraints() });
  } catch (e) {
    addMsg('err', 'Camera unavailable (' + e.name + '); continuing with audio only.');
    camOn = false;
    return await navigator.mediaDevices.getUserMedia({ audio });
  }
}

/** Start a conversation: acquire media, build the mic and playback audio graphs, open the socket, start frame upload. */
async function start() {
  ending = false;
  $('start').disabled = true; setLabel($('start'), 'Starting…'); setConn('connecting'); $('info').textContent = 'Waiting for camera and microphone access: allow it in the browser prompt…';
  try { adoptCameraSettings(await (await fetch('/api/settings', { cache: 'no-store' })).json()); } catch (e) { /* keep defaults */ }
  try { stream = await getMedia(); } catch (e) { addMsg('err', 'Microphone access failed: ' + e.message); $('start').disabled = false; setLabel($('start'), 'Start conversation'); setConn(''); $('info').textContent = 'Not connected'; return; }
  const hasVideo = stream.getVideoTracks().length > 0;
  if (hasVideo) learnCameraCapabilities(stream.getVideoTracks()[0]);
  $('video').srcObject = stream; $('novideo').hidden = hasVideo;
  $('video').onloadedmetadata = $('video').onresize = () => { const video = $('video'); if (video.videoWidth) document.querySelector('.video-wrap').style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight; };
  if (!hasVideo) { $('cam').disabled = true; }

  micCtx = new AudioContext({ latencyHint: 'interactive' });
  await micCtx.audioWorklet.addModule('worklets.js');
  micNode = new AudioWorkletNode(micCtx, 'mic-processor');
  micCtx.createMediaStreamSource(new MediaStream(stream.getAudioTracks())).connect(micNode);
  micNode.port.onmessage = (event) => sendBin(1, event.data);  // worklet posts int16 16 kHz PCM

  playCtx = new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' });
  await playCtx.audioWorklet.addModule('worklets.js');
  playNode = new AudioWorkletNode(playCtx, 'player-processor');
  playNode.connect(playCtx.destination);
  playNode.port.onmessage = (event) => { if (event.data.playing === false) send({ type: 'playback', state: 'idle' }); };
  await Promise.all([micCtx.resume(), playCtx.resume()]);

  connect();
  if (hasVideo) frameTimer = setInterval(sendFrame, 250);
}

// Fully stop: releases camera + mic (browser indicator goes off), closes the socket and audio, resets the UI.
function endConversation() {
  ending = true;
  clearInterval(frameTimer);
  try { send({ type: 'interrupt' }); ws && ws.close(); } catch (e) {}
  if (stream) stream.getTracks().forEach((track) => track.stop());
  for (const audioContext of [micCtx, playCtx]) { try { audioContext && audioContext.close(); } catch (e) {} }
  micNode = playNode = micCtx = playCtx = stream = ws = null;
  $('video').srcObject = null; $('novideo').hidden = false;
  $('faces').textContent = ''; $('spk').hidden = true; $('seen').hidden = true;
  setEnabled(false); setPill('idle'); $('info').textContent = 'Conversation ended'; setConn(''); setCtx(null);
  micOn = camOn = speakOn = true;
  toggle($('mic'), true, 'Mic on', 'Mic muted'); toggle($('cam'), true, 'Camera on', 'Camera off'); toggle($('speak'), true, 'Voice on', 'Voice off');
  setLabel($('start'), 'Start conversation'); $('start').title = 'Start conversation'; $('start').classList.remove('danger');
  $('start').onclick = start; $('start').disabled = false;
}

/** Open the WebSocket and wire the UI to its lifecycle. */
function connect() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setEnabled(true);
    if (!stream.getVideoTracks().length) $('cam').disabled = true;
    setLabel($('start'), 'End conversation'); $('start').title = 'End conversation'; $('start').classList.add('danger'); setConn('live');
    $('start').onclick = endConversation; $('start').disabled = false;
  };
  ws.onclose = () => { if (ending) return; setEnabled(false); setPill('idle'); $('info').textContent = 'Disconnected'; setConn('error'); setLabel($('start'), 'Reconnect'); $('start').classList.remove('danger'); $('start').disabled = false; $('start').onclick = () => location.reload(); };
  ws.onmessage = onMessage;
}

/* ---------- Sending ---------- */
/** Send a JSON control message if the socket is open. */
function send(message) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(message)); }
/** Send a binary frame: one tag byte (1 = mic PCM, 2 = JPEG) followed by the payload. Dropped when the socket is backed up (>2 MB). */
function sendBin(tag, payload) {
  if (!ws || ws.readyState !== 1 || ws.bufferedAmount > 2e6) return;
  const frame = new Uint8Array(1 + payload.byteLength); frame[0] = tag; frame.set(new Uint8Array(payload), 1); ws.send(frame);
}

/** Grab the current video frame, downscale its longer side to the configured size (default 448 px), and send it as JPEG (skipped when camera off or tab hidden). */
function sendFrame() {
  const video = $('video');
  if (!camOn || !video.videoWidth || document.hidden) return;
  const scale = Math.min(1, cameraSettings.maxSide / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  frameContext.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => blob && blob.arrayBuffer().then((jpegBytes) => sendBin(2, jpegBytes)), 'image/jpeg', 0.7);
}

/* ---------- Receiving ---------- */
/** Handle a socket message: binary = TTS audio to play; text = JSON dispatched on its `type`. */
function onMessage(event) {
  if (typeof event.data !== 'string') {  // 0x03 + int16 24 kHz PCM
    const pcm16 = new Int16Array(event.data.slice(1));
    const pcmFloat = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) pcmFloat[i] = pcm16[i] / 32768;  // int16 -> [-1, 1)
    playNode.port.postMessage({ pcm: pcmFloat }, [pcmFloat.buffer]);  // transfer, not copy
    return;
  }
  const m = JSON.parse(event.data);
  switch (m.type) {
    case 'info': $('info').textContent = `${m.backend}: ${m.model} · STT ${m.stt} · TTS ${m.tts} · deep think: ${m.deep}`; $('info').title = $('info').textContent; $('backend').value = m.backend; for (const o of $('backend').options) { o.disabled = !m.available.includes(o.value); o.textContent = o.textContent.replace(' (offline)', '') + (o.disabled ? ' (offline)' : ''); } break;
    case 'state': setPill(m.value); break;
    case 'transcript': addMsg('user', m.text); currentBotBubble = null; break;
    case 'assistant_start': currentBotBubble = addMsg('bot', ''); currentBotBubble.classList.add('streaming'); scrollLog(); break;
    case 'assistant_delta': if (currentBotBubble) { const wasNearBottom = nearBottom(); currentBotBubble.textContent += m.text; if (wasNearBottom) $('log').scrollTop = 1e9; } break;
    case 'assistant_done': if (currentBotBubble) { currentBotBubble.classList.remove('streaming'); if (!currentBotBubble.textContent.trim()) currentBotBubble.parentElement.remove(); } break;
    case 'stop_audio': playNode.port.postMessage({ clear: true }); break;
    case 'vision':
      if (m.frames) { $('seen').hidden = false; $('seen').textContent = `${m.frames} frame${m.frames > 1 ? 's' : ''} sent`; clearTimeout(framesSentTimer); framesSentTimer = setTimeout(() => $('seen').hidden = true, 2500); }
      break;
    case 'latency':
      if (m.stt_ms != null) $('mStt').textContent = m.stt_ms + ' ms';
      if (m.llm_first_ms != null) $('mLlm').textContent = m.llm_first_ms + ' ms';
      if (m.first_audio_ms != null) $('mAudio').textContent = m.first_audio_ms + ' ms';
      break;
    case 'context': setCtx(m); break;
    case 'faces': drawFaces(m.faces); break;
    case 'speaker': {
      const el = $('spk');
      if (m.name) { el.hidden = false; el.textContent = `${m.name} · ${m.how === 'voice' ? 'voice' : 'face'}`; clearTimeout(speakerBadgeTimer); speakerBadgeTimer = setTimeout(() => el.hidden = true, 6000); }
      else { el.hidden = false; el.textContent = 'Unknown speaker'; clearTimeout(speakerBadgeTimer); speakerBadgeTimer = setTimeout(() => el.hidden = true, 4000); }
      break;
    }
    case 'people': renderPeople(m.people); break;
    case 'people_event': addEvent(m.kind, m.text); break;
    case 'skill_event': addEvent(m.kind, m.text, m.detail); break;
    case 'error': addMsg('err', m.text); break;
  }
}

/* ---------- Small UI helpers ---------- */
/** Set the text of a button's `.label` child. */
function setLabel(btn, text) { btn.querySelector('.label').textContent = text; }
/** Reflect an on/off state on a toggle button: label text, `off` class and aria-pressed. */
function toggle(btn, on, labelOn, labelOff) { setLabel(btn, on ? labelOn : labelOff); btn.classList.toggle('off', !on); btn.setAttribute('aria-pressed', String(on)); }

/** Context meter: text readout + slim progress bar from a `context` message ({tokens, budget, summarized}); null resets. */
function setCtx(m) {
  const bar = $('ctxBar'), track = $('ctxTrack');
  if (!m) { $('ctx').textContent = 'Context: not started'; bar.style.width = '0%'; bar.className = ''; track.setAttribute('aria-valuenow', '0'); return; }
  const pct = m.budget ? Math.min(100, Math.round(m.tokens / m.budget * 100)) : 0;
  $('ctx').textContent = `Context ${(m.tokens/1000).toFixed(1)}k / ${(m.budget/1000).toFixed(0)}k tokens` + (m.summarized ? ' · older turns summarized' : '');
  $('ctx').title = $('ctx').textContent;
  bar.style.width = pct + '%'; bar.className = pct >= 90 ? 'hot' : pct >= 70 ? 'warn' : '';
  track.setAttribute('aria-valuenow', String(pct));
}

/* ---------- Control button handlers ---------- */
$('start').onclick = start;
$('mic').onclick = () => { micOn = !micOn; micNode.port.postMessage({ muted: !micOn }); toggle($('mic'), micOn, 'Mic on', 'Mic muted'); };
$('cam').onclick = () => { camOn = !camOn; stream.getVideoTracks().forEach((track) => track.enabled = camOn); $('novideo').hidden = camOn; toggle($('cam'), camOn, 'Camera on', 'Camera off'); };
$('speak').onclick = () => { speakOn = !speakOn; send({ type: 'config', speak: speakOn }); if (!speakOn) playNode.port.postMessage({ clear: true }); toggle($('speak'), speakOn, 'Voice on', 'Voice off'); };
$('stop').onclick = () => { send({ type: 'interrupt' }); playNode.port.postMessage({ clear: true }); };
$('backend').onchange = (event) => send({ type: 'config', backend: event.target.value });
$('form').onsubmit = (event) => { event.preventDefault(); const typed = $('text').value.trim(); if (typed) { send({ type: 'text', text: typed }); $('text').value = ''; } };

/* ---------- Faces and people ---------- */
/** Draw face boxes ({box: [x, y, w, h] as 0..1 fractions, name, speaking}) over the video.
 * The preview is shown as the camera sees it (not mirrored), so x maps straight across. */
function drawFaces(list) {
  const root = $('faces'); root.textContent = '';
  for (const face of list || []) {
    const [x, y, w, h] = face.box;
    const boxEl = document.createElement('div');
    boxEl.className = 'fbox' + (face.name ? ' known' : '') + (face.speaking ? ' speaking' : '');
    const left = x;
    boxEl.style.left = (left * 100) + '%'; boxEl.style.top = (y * 100) + '%';
    boxEl.style.width = (w * 100) + '%'; boxEl.style.height = (h * 100) + '%';
    const label = document.createElement('span'); label.className = 'flabel';
    label.textContent = (face.name || 'unknown') + (face.speaking ? ' · speaking' : '');
    if (left > 0.5) label.classList.add('r');   // keep labels inside the frame
    if (y < 0.2 && (left < 0.3 || left + w > 0.7) && y + h < 0.8) label.classList.add('below');   // clear of the state pill / frame-sent chip
    else if (y < 0.08) label.classList.add('in');
    boxEl.appendChild(label); root.appendChild(boxEl);
  }
}

// Set of 'name|faceCount|voiceCount' keys from the previous render; chips whose key is new get a flash animation.
let previousPeopleKeys = null;
/** Render the enrolled-people chips from a `people` message ([{name, face, voice}]), each with a forget button. */
function renderPeople(list) {
  const root = $('people'); root.textContent = '';
  const keyOf = (person) => `${person.name}|${person.face}|${person.voice}`;
  const previousKeys = previousPeopleKeys; previousPeopleKeys = new Set((list || []).map(keyOf));
  if (!list || !list.length) { const note = document.createElement('span'); note.className = 'empty-note'; note.textContent = 'Nobody enrolled yet'; root.appendChild(note); return; }
  for (const person of list) {
    const chip = document.createElement('span'); chip.className = 'chip' + (previousKeys && !previousKeys.has(keyOf(person)) ? ' flash' : '');
    const avatar = document.createElement('span'); avatar.className = 'avatar'; avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = (String(person.name).trim().split(/\s+/).map((word) => word[0]).slice(0, 2).join('') || '?').toUpperCase();  // initials
    const who = document.createElement('span'); who.className = 'who';
    const nameEl = document.createElement('b'); nameEl.textContent = person.name; chip.title = person.name;
    const samplesEl = document.createElement('small'); samplesEl.textContent = (person.face || person.voice) ? `${person.face ? 'face' : ''}${person.face && person.voice ? ' + ' : ''}${person.voice ? 'voice' : ''}` : 'not enrolled';
    who.append(nameEl, samplesEl);
    const forgetBtn = document.createElement('button'); forgetBtn.type = 'button'; forgetBtn.textContent = '×'; forgetBtn.title = 'Forget ' + person.name; forgetBtn.setAttribute('aria-label', 'Forget ' + person.name);
    forgetBtn.onclick = () => send({ type: 'forget', name: person.name });
    chip.append(avatar, who, forgetBtn); root.appendChild(chip);
  }
}

// [icon, caption] per people_event kind. "failed" means the backend did NOT change anything.
const EVENT_STYLES = {
  done: ['✓', ''], learned: ['✓', ''], renamed: ['✓', ''], forgot: ['✓', ''],
  needs_input: ['?', 'Waiting for your answer'], failed: ['✕', 'Nothing was changed'], info: ['i', ''], working: ['…', 'Working on it'],
};
/** Add a live outcome row (naming / renaming / forgetting attempt) to the chat log. */
function addEvent(kind, text, detail = []) {
  const [icon, caption] = EVENT_STYLES[kind] || EVENT_STYLES.info;
  const stick = nearBottom();
  $('empty').hidden = true;
  const row = document.createElement('div'); row.className = 'evt ' + kind; row.setAttribute('role', 'status');
  const iconEl = document.createElement('span'); iconEl.className = 'evt-icon'; iconEl.setAttribute('aria-hidden', 'true'); iconEl.textContent = icon;
  const body = document.createElement('span'); body.className = 'evt-body';
  const textEl = document.createElement('span'); textEl.textContent = text; body.appendChild(textEl);
  if (caption) { const captionEl = document.createElement('small'); captionEl.textContent = caption; body.appendChild(captionEl); }
  if (detail.length) {   // full list (e.g. every file name), shown in a scrollable box
    row.classList.add('has-detail');
    const list = document.createElement('ul'); list.className = 'evt-detail';
    for (const line of detail) { const item = document.createElement('li'); item.textContent = line; list.appendChild(item); }
    body.appendChild(list);
  }
  row.append(iconEl, body); $('log').appendChild(row);
  if (stick) $('log').scrollTop = 1e9;
}

/* ---------- Settings dialog ---------- */
// dlg: the <dialog>; msg: status line; audio: the preview Audio element currently playing (or null).
const cfg = { dlg: $('cfg'), msg: $('cfgMsg'), audio: null };
/** Show a status line in the settings dialog (cls: '' or 'err'). */
const cfgMsg = (text, cls = '') => { cfg.msg.textContent = text; cfg.msg.className = 'cfgmsg ' + cls; };
/** Rebuild a <select> from [value, label] pairs and select `selected`. */
function fillSelect(select, options, selected) {
  select.textContent = '';
  for (const [value, label] of options) select.add(new Option(label, value));
  select.value = selected;
}
/** Build one checkbox per skill (label, description) and tick the enabled ones. */
function renderSkillToggles(catalog, enabled) {
  const root = $('cfgSkills'); root.textContent = '';
  for (const skill of catalog) {
    const label = document.createElement('label'); label.className = 'skill-row';
    const box = document.createElement('input'); box.type = 'checkbox'; box.value = skill.id; box.checked = enabled.includes(skill.id);
    const text = document.createElement('span'); text.textContent = skill.label;
    const hint = document.createElement('small'); hint.textContent = skill.description; text.appendChild(hint);
    label.append(box, text); root.appendChild(label);
  }
}
/** Fill the dialog's inputs and their min/max from a /api/settings payload. */
/** Ask the server which models an endpoint offers; fills the suggestion list and says whether it is reachable. */
async function checkEndpoint(name) {
  const note = $(`cfg${name}Note`), list = $(`cfg${name}List`), model = $(`cfg${name}Model`);
  note.textContent = 'checking…'; note.className = '';
  try {
    const response = await fetch('/api/llm/models', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ url: $(`cfg${name}Url`).value }) });
    const result = await response.json();
    if (!result.ok) { note.textContent = '✕ ' + (result.error || 'not reachable'); note.className = 'bad'; return; }
    list.textContent = ''; for (const id of result.models) list.appendChild(new Option(id));
    const found = result.models.includes(model.value.trim());
    note.textContent = `✓ reachable, ${result.models.length} models` + (found ? '' : ' · this model name is not in the list'); note.className = found ? 'good' : 'bad';
  } catch (error) { note.textContent = '✕ could not check'; note.className = 'bad'; }
}
for (const button of document.querySelectorAll('button.check')) button.onclick = () => checkEndpoint(button.dataset.for);
/** Show the voices, speed control and note for the engine chosen in the dropdown (`voice` pre-selects one). */
function showEngine(voice) {
  const engine = cfg.engines.find((entry) => entry.id === $('cfgEngine').value) || cfg.engines[0];
  if (!engine) return;
  fillSelect($('cfgVoice'), engine.voices.map((entry) => [entry.id, entry.label]), engine.voices.some((entry) => entry.id === voice) ? voice : engine.voices[0].id);
  $('cfgSpeed').disabled = !engine.speed;
  $('cfgEngineNote').textContent = engine.note + (engine.speed ? '' : ' Speech speed is not adjustable.');
}
$('cfgEngine').onchange = () => showEngine();
const cfgFill = (settings) => {
  $('cfgOllamaUrl').value = settings.ollama_url; $('cfgOllamaModel').value = settings.ollama_model;
  $('cfgLmstudioUrl').value = settings.lmstudio_url; $('cfgLmstudioModel').value = settings.lmstudio_model;
  for (const name of ['Ollama', 'Lmstudio']) { $(`cfg${name}Note`).textContent = ''; $(`cfg${name}Note`).className = ''; }
  cfg.engines = settings.tts_engines || [];
  fillSelect($('cfgEngine'), cfg.engines.map((engine) => [engine.id, engine.label]), settings.tts_engine);
  showEngine(settings.tts_voice);
  fillSelect($('cfgVision'), (settings.vision_modes || []).map((mode) => [mode.id, mode.label]), settings.vision_mode); $('cfgCtx').value = settings.context_tokens; $('cfgKeep').value = settings.context_keep_tokens; 
  $('cfgCtx').min = settings.limits.context_tokens[0]; $('cfgCtx').max = settings.limits.context_tokens[1];
  $('cfgKeep').min = settings.limits.keep_min; $('cfgKeep').max = settings.limits.context_tokens[1];
  $('cfgCamRes').value = settings.camera_resolution; refreshCameraOptions();
  fillSelect($('cfgFrameSide'), (settings.frame_sides || [320, 448, 640, 896, 1280]).map((side) => [String(side), side + ' px']), String(settings.frame_max_side));
  renderSkillToggles(settings.skills || [], settings.skills_enabled || []);
  const speedInput = $('cfgSpeed'); speedInput.min = settings.limits.tts_speed[0]; speedInput.max = settings.limits.tts_speed[1]; speedInput.value = settings.tts_speed;
  $('cfgSpeedOut').textContent = settings.tts_speed + '×';
};
/** Read the dialog's inputs as the JSON body expected by /api/settings and /api/tts/preview. */
const cfgValues = () => ({ tts_engine: $('cfgEngine').value, ollama_url: $('cfgOllamaUrl').value, ollama_model: $('cfgOllamaModel').value, lmstudio_url: $('cfgLmstudioUrl').value, lmstudio_model: $('cfgLmstudioModel').value, context_tokens: +$('cfgCtx').value, context_keep_tokens: +$('cfgKeep').value, tts_voice: $('cfgVoice').value, tts_speed: +$('cfgSpeed').value, skills_enabled: [...document.querySelectorAll('#cfgSkills input:checked')].map((box) => box.value), camera_resolution: $('cfgCamRes').value, vision_mode: $('cfgVision').value, frame_max_side: +$('cfgFrameSide').value });
/** Stop any preview audio and close the dialog. */
const cfgClose = () => { if (cfg.audio) { cfg.audio.pause(); cfg.audio = null; } cfg.dlg.close(); };
/* Collapsible sections: which are open is remembered, and each header shows the current value so closed sections stay informative. */
const CFG_OPEN_KEY = 'cfgOpenSections';
const cfgSections = [...document.querySelectorAll('#cfgForm details.sec')];
function loadOpenSections() {
  let open = ['llm'];            // first run: only the language model is expanded
  try { const saved = JSON.parse(localStorage.getItem(CFG_OPEN_KEY)); if (Array.isArray(saved)) open = saved; } catch (e) {}
  for (const section of cfgSections) section.open = open.includes(section.dataset.sec);
}
for (const section of cfgSections) section.addEventListener('toggle', () => {
  try { localStorage.setItem(CFG_OPEN_KEY, JSON.stringify(cfgSections.filter((item) => item.open).map((item) => item.dataset.sec))); } catch (e) {}
});
/** Refresh the one-line summary shown in each collapsed section header from the current form values. */
function updateSectionSummaries() {
  const selected = (id) => { const select = $(id); return select.selectedOptions[0] ? select.selectedOptions[0].textContent : ''; };
  const engineName = selected('cfgEngine').split(' (')[0];
  const text = {
    llm: `${$('cfgOllamaModel').value}`,
    context: `${Number($('cfgCtx').value || 0).toLocaleString()} tokens`,
    camera: `${$('cfgCamRes').value || ''} · ${$('cfgFrameSide').value} px`,
    voice: `${engineName} · ${selected('cfgVoice').split(' (')[0]}`,
    skills: `${document.querySelectorAll('#cfgSkills input:checked').length} on`,
  };
  for (const [key, value] of Object.entries(text)) { const el = $('sum_' + key); if (el) el.textContent = value; }
}
// A field that fails the browser's own validation inside a closed section cannot show its message: open the section first.
$('cfgForm').addEventListener('invalid', (event) => { const section = event.target.closest('details.sec'); if (section) section.open = true; }, true);
$('cfgForm').addEventListener('input', updateSectionSummaries);
$('cfgForm').addEventListener('change', updateSectionSummaries);

$('cfgBtn').onclick = async () => {
  cfgMsg('Loading…');
  loadOpenSections();
  try { const response = await fetch('/api/settings', { cache: 'no-store' }); cfgFill(await response.json()); updateSectionSummaries(); cfgMsg(''); }
  catch (e) { cfgMsg('Could not load settings: ' + e.message, 'err'); }
  cfg.dlg.showModal();
};
$('cfgClose').onclick = $('cfgCancel').onclick = cfgClose;
cfg.dlg.addEventListener('click', (event) => { if (event.target === cfg.dlg) cfgClose(); });  // click on the backdrop closes
$('cfgSpeed').oninput = () => { $('cfgSpeedOut').textContent = $('cfgSpeed').value + '×'; };
$('cfgPreview').onclick = async () => {
  const previewBtn = $('cfgPreview'); previewBtn.disabled = true; cfgMsg($('cfgEngine').value === 'kokoro' ? 'Generating preview…' : 'Generating preview… (the first time a new engine loads, it may take a while)');
  try {
    const response = await fetch('/api/tts/preview', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(cfgValues()) });
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error((body.errors || [response.statusText]).join('; ')); }
    if (cfg.audio) cfg.audio.pause();
    cfg.audio = new Audio(URL.createObjectURL(await response.blob())); await cfg.audio.play(); cfgMsg('');  // response is a WAV
  } catch (e) { cfgMsg('Preview failed: ' + e.message, 'err'); }
  previewBtn.disabled = false;
};
$('cfgForm').onsubmit = async (e) => {
  e.preventDefault(); const saveBtn = $('cfgSave'); saveBtn.disabled = true; cfgMsg($('cfgEngine').value === 'kokoro' ? 'Saving…' : 'Saving… (loading the voice engine; the first time it downloads the model)');
  try {
    const response = await fetch('/api/settings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(cfgValues()) });
    const result = await response.json();
    if (!response.ok || !result.ok) { cfgMsg((result.errors || ['Save failed']).join('. '), 'err'); saveBtn.disabled = false; return; }
    cfgFill(result);
    adoptCameraSettings(result);
    const liveTrack = stream && stream.getVideoTracks()[0];
    if (liveTrack) liveTrack.applyConstraints(videoConstraints()).catch((error) => addMsg('err', 'Camera could not switch resolution: ' + error.message));
    if (result.saved === false) { cfgMsg(result.warning, 'err'); saveBtn.disabled = false; return; }  // applied but not persisted
    addEvent && ws && ws.readyState === 1 && addEvent('info', 'Settings saved');
    cfgClose();
  } catch (err) { cfgMsg('Save failed: ' + err.message, 'err'); }
  saveBtn.disabled = false;
};

// Read the camera's limits as soon as the page loads (only if permission was already granted) and on hot-plug.
refreshCameraOptions();   // never leave the dropdown empty, even before the camera is detected
probeCamera();
if (navigator.mediaDevices) navigator.mediaDevices.addEventListener('devicechange', probeCamera);
