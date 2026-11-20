// Mic: box-filter downsample to 16 kHz, emit 512-sample (32 ms) Int16 frames.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0; this.cnt = 0; this.phase = 0;
    this.out = new Int16Array(512); this.n = 0;
    this.muted = false;
    this.port.onmessage = (e) => { if ('muted' in e.data) this.muted = e.data.muted; };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; this.cnt++; this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.cnt));
        this.acc = 0; this.cnt = 0;
        this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
        if (this.n === 512) {
          if (!this.muted) { const b = this.out.slice(); this.port.postMessage(b.buffer, [b.buffer]); }
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);

// Player: gapless queue of Float32 chunks; instant flush for barge-in.
class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = []; this.cur = null; this.pos = 0; this.playing = false;
    this.port.onmessage = (e) => {
      if (e.data.clear) { this.q = []; this.cur = null; this.pos = 0; this.notify(false); }
      else if (e.data.pcm) { this.q.push(e.data.pcm); }
    };
  }
  notify(p) { if (p !== this.playing) { this.playing = p; this.port.postMessage({ playing: p }); } }
  process(_, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      if (!this.cur || this.pos >= this.cur.length) { this.cur = this.q.shift() || null; this.pos = 0; }
      out[i] = this.cur ? this.cur[this.pos++] : 0;
    }
    this.notify(!!this.cur);
    return true;
  }
}
registerProcessor('player-processor', PlayerProcessor);
