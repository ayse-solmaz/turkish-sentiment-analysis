// Transformers.js: Hugging Face modellerini tarayıcıda ONNX Runtime Web ile çalıştırır.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

// Kendi bilgisayarımızda (localhost) modeli /models/ klasöründen, canlıda Hugging Face'ten yükle.
const LOCAL = ['localhost', '127.0.0.1'].includes(location.hostname);
const HF_MODEL = 'KULLANICI_ADINIZ/turkish-sentiment-electra-small';
const MODEL = LOCAL ? 'electra-sentiment' : HF_MODEL;
env.allowLocalModels = LOCAL;
env.allowRemoteModels = !LOCAL;
env.localModelPath = '/models/';

const LABELS = {
  Positive: { name: 'Olumlu', color: 'var(--pos)' },
  Notr: { name: 'Nötr', color: 'var(--neu)' },
  Negative: { name: 'Olumsuz', color: 'var(--neg)' },
};

const $ = (id) => document.getElementById(id);
let classifier = null;

// "webgpu|fp32" -> { device: 'webgpu', dtype: 'fp32' }
// dtype 'fp32' -> onnx/model.onnx,  'q8' -> onnx/model_quantized.onnx
function parse(value) {
  const [device, dtype] = value.split('|');
  return { device, dtype };
}

async function load(value) {
  const { device, dtype } = parse(value);
  const t = performance.now();
  const model = await pipeline('text-classification', MODEL, { device, dtype });
  return { model, ms: performance.now() - t };
}

// ---------- Analiz ----------
async function switchBackend() {
  $('analyze').disabled = true;
  $('status').textContent = 'Model yükleniyor…';
  if (classifier) await classifier.dispose();
  classifier = null;
  try {
    const { model, ms } = await load($('backend').value);
    classifier = model;
    $('status').textContent = `Hazır (${ms.toFixed(0)} ms'de yüklendi)`;
    $('analyze').disabled = false;
    analyze();
  } catch (err) {
    console.error(err);
    $('status').textContent = `Bu ortam çalışmadı: ${err.message}`;
  }
}

async function analyze() {
  const text = $('text').value.trim();
  if (!classifier || !text) return;
  const t = performance.now();
  const scores = await classifier(text, { top_k: null }); // top_k: null -> tüm sınıfların skorları
  $('latency').textContent = `${(performance.now() - t).toFixed(1)} ms`;

  $('result').innerHTML = scores
    .map(({ label, score }, i) => `
      <div class="bar ${i === 0 ? 'top' : ''}">
        <div class="bar-label"><span>${LABELS[label].name}</span><span>${(score * 100).toFixed(1)}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${score * 100}%;background:${LABELS[label].color}"></div></div>
      </div>`)
    .join('');
}

// ---------- Hız karşılaştırması ----------
const WARMUP = 5;
const RUNS = 50;

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

async function benchmark() {
  $('bench').disabled = true;
  $('analyze').disabled = true;
  const tbody = $('benchTable').querySelector('tbody');
  tbody.innerHTML = '';
  $('benchTable').hidden = false;
  const text = $('text').value.trim() || 'Kargo çok hızlı geldi, ürün harika!';

  for (const option of $('backend').options) {
    const row = tbody.insertRow();
    row.innerHTML = `<td>${option.text}</td><td colspan="4">ölçülüyor…</td>`;
    try {
      const { model, ms: loadMs } = await load(option.value);

      // İlk çalıştırma ayrı ölçülür: WebGPU burada shader'ları derler, bu yüzden yavaştır.
      let t = performance.now();
      await model(text);
      const firstMs = performance.now() - t;

      for (let i = 0; i < WARMUP; i++) await model(text);

      const times = [];
      for (let i = 0; i < RUNS; i++) {
        t = performance.now();
        await model(text); // await: sonuç gerçekten hazır olana kadar bekle (GPU dahil)
        times.push(performance.now() - t);
      }
      times.sort((a, b) => a - b);
      await model.dispose();

      row.innerHTML = `<td>${option.text}</td><td>${loadMs.toFixed(0)} ms</td><td>${firstMs.toFixed(1)} ms</td>
        <td><b>${percentile(times, 0.5).toFixed(1)} ms</b></td><td>${percentile(times, 0.95).toFixed(1)} ms</td>`;
    } catch (err) {
      console.error(err);
      row.innerHTML = `<td>${option.text}</td><td colspan="4">çalışmadı: ${err.message}</td>`;
    }
  }
  $('bench').disabled = false;
  $('analyze').disabled = !classifier;
}

// ---------- Başlangıç ----------
// WebGPU desteklenmiyorsa o seçeneği kapat.
if (!navigator.gpu) {
  const webgpu = $('backend').querySelector('option[value^="webgpu"]');
  webgpu.disabled = true;
  webgpu.text += ' (bu tarayıcıda yok)';
}

$('backend').addEventListener('change', switchBackend);
$('analyze').addEventListener('click', analyze);
$('text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
});
$('bench').addEventListener('click', benchmark);

switchBackend();
