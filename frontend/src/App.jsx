import { useEffect, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const LABELS = [
  { key: "PRODUK", name: "Produk" },
  { key: "JUMLAH", name: "Jumlah" },
  { key: "NAMA_PERSONALISASI", name: "Nama personalisasi" },
  { key: "WARNA", name: "Warna" },
  { key: "UKURAN", name: "Ukuran" },
  { key: "TENGGAT", name: "Tenggat" },
  { key: "PENERIMA_ACARA", name: "Penerima / acara" },
  { key: "TEKS_KARTU", name: "Teks kartu" },
];
const NAME_OF = Object.fromEntries(LABELS.map((l) => [l.key, l.name]));
const ORDER_TYPES = ["Ready Stock", "Custom Sajadah", "Custom Handuk", "Hampers", "Other"];

const EXAMPLES = [
  "halo kak, mau pesan sajadah travel 2 pcs nama Aisyah warna navy ukuran 70x140 butuh tanggal 20 november",
  'kak mau pesan hampers lebaran 10 set untuk ibu kartunya "Selamat Idul Fitri"',
  "min, sjadah bordir 2 pc nm Dewi wrn maroon tgl 5 des",
  "kak mau pesan sajadah 2 pcs",
];

async function post(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail; } catch { /* abaikan */ }
    if (detail && typeof detail !== "string") detail = JSON.stringify(detail);
    throw new Error(detail || `Server error ${res.status}`);
  }
  return res.json();
}

function Highlighted({ text, spans }) {
  const parts = [];
  let pos = 0;
  [...spans].sort((a, b) => a.start - b.start).forEach((s, i) => {
    if (s.start < pos) return;
    if (s.start > pos) parts.push(<span key={`t${i}`}>{text.slice(pos, s.start)}</span>);
    parts.push(
      <mark
        key={`m${i}`}
        className={`hl hl-${s.label} ${s.low_confidence ? "low" : ""}`}
        title={`${NAME_OF[s.label]} | confidence ${s.confidence}`}
      >
        {text.slice(s.start, s.end)}
        <sup>{NAME_OF[s.label]}</sup>
      </mark>
    );
    pos = s.end;
  });
  if (pos < text.length) parts.push(<span key="end">{text.slice(pos)}</span>);
  return <p className="highlighted">{parts}</p>;
}

const CATEGORY_NAME = {
  sajadah: "Sajadah", handuk: "Handuk", mug: "Mug & tumbler",
  hampers: "Hampers & kado", mukena: "Mukena",
};

function Chips({ items }) {
  return (
    <div className="chips">
      {items.map((x) => <span key={x} className="chip">{x}</span>)}
    </div>
  );
}

function Catalog({ data, error }) {
  return (
    <details className="card catalog" open>
      <summary>Katalog tersedia</summary>
      {error && <p className="hint">{error}</p>}
      {data && (
        <>
          <p className="hint">{data.catatan}</p>

          <h3>Produk</h3>
          {Object.entries(data.produk).map(([cat, items]) => (
            <div key={cat} className="cat-row">
              <b>{CATEGORY_NAME[cat] || cat}</b>
              <Chips items={items} />
            </div>
          ))}

          <h3>Warna</h3>
          <Chips items={data.warna} />

          <h3>Ukuran</h3>
          <Chips items={[...data.ukuran.huruf, ...data.ukuran.dimensi]} />
          <p className="hint">
            Berlaku untuk: {data.ukuran.berlaku_untuk.map((c) => CATEGORY_NAME[c] || c).join(", ")}
          </p>

          <h3>Satuan jumlah</h3>
          <Chips items={data.satuan_jumlah} />

          <h3>Field wajib per produk</h3>
          <ul className="req">
            <li>Semua produk: {data.field_wajib.semua_produk.map((f) => NAME_OF[f]).join(", ")}</li>
            {Object.entries(data.field_wajib)
              .filter(([k]) => k !== "semua_produk")
              .map(([cat, fs]) => (
                <li key={cat}>
                  {CATEGORY_NAME[cat] || cat}: tambahan {fs.map((f) => NAME_OF[f]).join(", ")}
                </li>
              ))}
          </ul>
        </>
      )}
    </details>
  );
}

const NUMBER_WORDS = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan",
  "sembilan", "sepuluh", "sebelas", "belas", "puluh", "ratus", "lusin", "setengah"];

const splitValues = (v) => (v || "").split(";").map((s) => s.trim()).filter(Boolean);
const normSize = (s) => s.toLowerCase().replace(/\s+/g, "").replace(/cm$/, "");

/** Nilai pada sebuah field yang TIDAK cocok dengan katalog (hanya PRODUK, WARNA, UKURAN, JUMLAH). */
function outOfCatalog(key, value, catalog) {
  if (!catalog) return [];
  const items = splitValues(value);

  if (key === "PRODUK") {
    const terms = Object.values(catalog.produk).flat().map((t) => t.toLowerCase());
    return items.filter((v) => {
      const l = v.toLowerCase();
      return !terms.some((t) => l.includes(t) || (l.length >= 4 && t.includes(l)));
    });
  }
  if (key === "WARNA") {
    const colors = catalog.warna.map((c) => c.toLowerCase());
    return items.filter((v) => !colors.includes(v.toLowerCase()));
  }
  if (key === "UKURAN") {
    const ok = new Set([...catalog.ukuran.huruf, ...catalog.ukuran.dimensi].map(normSize));
    return items.filter((v) => !ok.has(normSize(v)));
  }
  if (key === "JUMLAH") {
    const units = catalog.satuan_jumlah;
    return items.filter((v) => {
      const l = v.toLowerCase();
      return !(/\d/.test(l) || NUMBER_WORDS.some((w) => l.includes(w)) || units.some((u) => l.includes(u)));
    });
  }
  return [];   // nama, tenggat, penerima/acara, teks kartu: nilainya terbuka
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [form, setForm] = useState({});
  const [jenis, setJenis] = useState("");
  const [missing, setMissing] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [error, setError] = useState("");
  const [catalog, setCatalog] = useState(null);
  const [catalogError, setCatalogError] = useState("");
  const [corrections, setCorrections] = useState({});

  useEffect(() => {
    fetch(`${API}/catalog`)
      .then((r) => {
        if (!r.ok) throw new Error(`Server error ${r.status}`);
        return r.json();
      })
      .then(setCatalog)
      .catch(() => setCatalogError("Katalog tidak dapat dimuat (backend belum berjalan?)"));
  }, []);

  function explain(e, action) {
    return e instanceof TypeError
      ? `Gagal ${action}: API tidak terjangkau di ${API}. Pastikan backend (uvicorn) sedang berjalan.`
      : `Gagal ${action}: ${e.message}`;
  }

  async function handleExtract() {
    setError("");
    setSummary(null);
    setLoading(true);
    try {
      const r = await post("/extract", { text });
      setResult(r);
      setJenis(r.jenis_pesanan || "");
      const f = {};
      const corr = {};
      LABELS.forEach(({ key }) => {
        const items = r.normalisasi?.[key] || [];
        const fixed = items.filter((i) => i.normal);
        if (fixed.length > 0) {
          f[key] = items.map((i) => i.normal || i.asli).join("; ");
          corr[key] = fixed;
        } else {
          f[key] = (r.entities[key] || []).join("; ");
        }
      });
      setForm(f);
      setCorrections(corr);
      setMissing(r.missing_fields);
    } catch (e) {
      setResult(null);
      setError(explain(e, "mengekstrak"));
    } finally {
      setLoading(false);
    }
  }

  function revertCorrection(key) {
    const items = result?.normalisasi?.[key] || [];
    setForm({ ...form, [key]: items.map((i) => i.asli).join("; ") });
    setCorrections({ ...corrections, [key]: [] });
  }

  async function handleSummarize() {
    setError("");
    setSummarizing(true);
    try {
      const entities = {};
      LABELS.forEach(({ key }) => {
        entities[key] = (form[key] || "").split(";").map((s) => s.trim()).filter(Boolean);
      });
      const r = await post("/summarize", { jenis_pesanan: jenis || null, entities });
      setSummary(r);
      setMissing(r.missing_fields);
    } catch (e) {
      setError(explain(e, "membuat ringkasan"));
    } finally {
      setSummarizing(false);
    }
  }

  const lowFields = new Set(result?.low_confidence ?? []);
  const missingNow = missing.filter((f) => !(form[f] || "").trim());

  return (
    <div className="app">
      <header>
        <h1>Asisten Input Pesanan Firaza</h1>
        <p>Tempel pesan customer, periksa hasil ekstraksi, lalu buat ringkasan pesanan.</p>
      </header>

      <div className="grid">
        <section className="card">
          <h2>1. Pesan customer</h2>
          <textarea
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Tempel pesan customer di sini..."
          />
          <div className="row">
            <button onClick={handleExtract} disabled={loading || !text.trim()}>
              {loading ? "Memproses..." : "Ekstrak pesanan"}
            </button>
            <span className="hint">Contoh:</span>
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="ghost" onClick={() => setText(ex)}>#{i + 1}</button>
            ))}
          </div>
          {error && <div className="alert error">{error}</div>}

          {result && (
            <>
              <h2>Teks dengan sorotan entitas</h2>
              <Highlighted text={result.text} spans={result.spans} />
              <p className="hint">
                Garis putus-putus = model kurang yakin, mohon dicek.
                Model: {result.model} | confidence kalimat: {result.sentence_confidence}
              </p>
            </>
          )}
        </section>

        <section className="card">
          <h2>2. Form pesanan (bisa diedit)</h2>
          {!result ? (
            <p className="hint">Hasil ekstraksi akan muncul di sini.</p>
          ) : (
            <>
              <label className="field">
                <span>Jenis pesanan {result.jenis_confidence != null &&
                  <em>(keyakinan {result.jenis_confidence})</em>}</span>
                <select value={jenis} onChange={(e) => setJenis(e.target.value)}>
                  <option value="">(belum ditentukan)</option>
                  {ORDER_TYPES.map((t) => <option key={t}>{t}</option>)}
                </select>
              </label>

              {LABELS.map(({ key, name }) => {
                const isMissing = missingNow.includes(key);
                const isLow = lowFields.has(key) && (form[key] || "").trim();
                const offItems = outOfCatalog(key, form[key], catalog);
                const fixes = corrections[key] || [];
                return (
                  <label key={key} className="field">
                    <span>
                      {name}
                      {isMissing && <b className="badge red">wajib, kosong</b>}
                      {!isMissing && isLow && <b className="badge amber">cek</b>}
                      {offItems.length > 0 && <b className="badge gray">di luar katalog</b>}
                    </span>
                    <input
                      className={isMissing ? "bad" : (isLow || offItems.length > 0) ? "warn" : ""}
                      value={form[key] || ""}
                      onChange={(e) => {
                        setForm({ ...form, [key]: e.target.value });
                        setCorrections({ ...corrections, [key]: [] });
                      }}
                      placeholder="(tidak ditemukan)"
                    />
                    {fixes.length > 0 && (
                      <small className="fixed">
                        Dikoreksi otomatis: {fixes.map((c) => `${c.asli} → ${c.normal}`).join(", ")}{" "}
                        <button type="button" className="linklike" onClick={() => revertCorrection(key)}>
                          pakai teks asli
                        </button>
                      </small>
                    )}
                    {offItems.length > 0 && (
                      <small className="offcat">Tidak ada di katalog: {offItems.join(", ")}</small>
                    )}
                  </label>
                );
              })}
              <p className="hint">Beberapa nilai pada satu field dipisah titik koma (;).</p>

              <button onClick={handleSummarize} disabled={summarizing}>
                {summarizing ? "Membuat ringkasan..." : "Buat ringkasan pesanan"}
              </button>

              {summary && (
                <div className="summary">
                  <h3>Ringkasan</h3>
                  <p>{summary.summary}</p>
                  <p className="hint">
                    Sumber: {summary.source}
                    {summary.note ? ` | ${summary.note}` : ""}
                  </p>
                </div>
              )}
            </>
          )}
        </section>
      </div>
      <Catalog data={catalog} error={catalogError} />
    </div>
  );
}