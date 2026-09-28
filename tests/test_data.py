"""Step 2 gate: clean speech, synthetic fallback and noise sources."""

import logging

import numpy as np
import pytest
import soundfile as sf

import config
from src.data import (
    choose_talkers,
    extract_clean_zip,
    load_audio_file,
    load_dataset,
    load_noizeus,
    load_speakers,
    make_babble,
    synthetic_pool,
    synthetic_speech,
    white_noise,
)

HAVE_NOIZEUS = any(config.CLEAN_DIR.glob("*.wav"))


@pytest.mark.skipif(not HAVE_NOIZEUS, reason="NOIZEUS not downloaded")
def test_load_noizeus_real():
    pool = load_noizeus()
    assert len(pool) == config.N_NOIZEUS_SENTENCES
    assert [n for n, _ in pool] == [f"sp{i:02d}" for i in range(1, 31)]
    for _, x in pool:
        assert x.ndim == 1 and x.dtype == np.float64
        assert 2.0 * config.FS < x.size < 4.0 * config.FS
        assert np.max(np.abs(x)) < 1.0


def test_load_noizeus_rejects_stereo(tmp_path):
    sf.write(tmp_path / "st.wav", np.zeros((800, 2)), config.FS)
    with pytest.raises(ValueError, match="mono"):
        load_noizeus(tmp_path)


def test_load_noizeus_resamples_to_8k(tmp_path, caplog):
    fs_in = 16000
    t = np.arange(fs_in) / fs_in
    sf.write(tmp_path / "hi.wav", 0.5 * np.sin(2 * np.pi * 500 * t), fs_in, subtype="FLOAT")
    with caplog.at_level(logging.INFO):
        ((name, x),) = load_noizeus(tmp_path)
    assert name == "hi"
    assert x.size == config.FS
    assert "resampled 16000 Hz -> 8000 Hz" in caplog.text
    # Away from the filter edges, the result is the same 500 Hz sine at 8 kHz.
    t8 = np.arange(config.FS) / config.FS
    mid = slice(500, config.FS - 500)
    np.testing.assert_allclose(x[mid], 0.5 * np.sin(2 * np.pi * 500 * t8)[mid], atol=1e-3)


@pytest.mark.skipif(not config.SPEAKERS_CSV.exists(), reason="speakers.csv missing")
def test_load_speakers():
    spk = load_speakers()
    assert len(spk) == 30
    speakers = {s for s, _ in spk.values()}
    assert len(speakers) == 6
    genders = {s: g for s, g in spk.values()}
    assert sorted(genders.values()).count("M") == 3
    assert sorted(genders.values()).count("F") == 3


def test_synthetic_speech_properties():
    x = synthetic_speech(config.FS, 3.0, np.random.default_rng(1))
    assert x.shape == (3 * config.FS,) and x.dtype == np.float64
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) == pytest.approx(config.SYNTH_PEAK, abs=1e-15)
    lead = int(config.SYNTH_LEAD_SILENCE_S * config.FS)
    assert np.all(x[:lead] == 0.0)
    # Speech-like: active syllables separated by exact-zero pauses.
    frames = x[: x.size // 256 * 256].reshape(-1, 256)
    energy = np.sum(frames**2, axis=1)
    assert np.sum(energy == 0) >= 2 and np.sum(energy > 0) >= 0.5 * frames.shape[0]


def test_synthetic_speech_spectrum_below_3k5():
    x = synthetic_speech(config.FS, 3.0, np.random.default_rng(2))
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / config.FS)
    # Voiced harmonics stop at 3.5 kHz; only fricative/burst noise is above.
    assert spec[f < 3500].sum() > 0.9 * spec.sum()


def test_synthetic_speech_seeded():
    a = synthetic_speech(rng=np.random.default_rng(5))
    b = synthetic_speech(rng=np.random.default_rng(5))
    c = synthetic_speech(rng=np.random.default_rng(6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_synthetic_pool_distinct():
    pool = synthetic_pool(4)
    assert [n for n, _ in pool] == ["syn01", "syn02", "syn03", "syn04"]
    sigs = [s for _, s in pool]
    assert all(not np.array_equal(sigs[0], s) for s in sigs[1:])


def test_load_dataset_falls_back_to_synthetic(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        ds = load_dataset(folder=tmp_path)
    assert ds.synthetic and ds.speakers is None
    assert len(ds.signals) == config.SYNTH_N_SENTENCES
    assert "synthetic" in caplog.text
    forced = load_dataset(force_synthetic=True)
    assert forced.synthetic and forced.names[0] == "syn01"


def test_white_noise():
    d = white_noise(100_000, np.random.default_rng(0))
    assert d.shape == (100_000,)
    assert abs(d.mean()) < 0.02 and abs(d.var() - 1.0) < 0.02


def test_choose_talkers_never_includes_target():
    for i in range(30):
        for seed in range(20):
            idx = choose_talkers(30, i, np.random.default_rng(seed))
            assert i not in idx
            assert idx.size == config.N_BABBLE_TALKERS == np.unique(idx).size


@pytest.mark.skipif(not config.SPEAKERS_CSV.exists(), reason="speakers.csv missing")
def test_choose_talkers_other_speakers_and_gender_mix():
    spk = load_speakers()
    speakers = [spk[f"sp{i:02d}"] for i in range(1, 31)]
    for i in range(30):
        idx = choose_talkers(30, i, np.random.default_rng(i), speakers=speakers)
        assert i not in idx
        assert all(speakers[j][0] != speakers[i][0] for j in idx)
        genders = [speakers[j][1] for j in idx]
        assert genders.count("M") == 3 and genders.count("F") == 3


def test_babble_for_sentence_i_contains_no_sentence_i():
    # Sentence j is a pure tone at 100*(j+1) Hz with a whole number of cycles
    # in 800 samples, so tiling is seamless and each tone is bin-exact in a
    # length-8000 FFT (1 Hz bins). Babble for i must have no energy at tone i.
    n_s, fs = 10, config.FS
    t = np.arange(800) / fs
    tones = [np.sin(2 * np.pi * 100 * (j + 1) * t) for j in range(n_s)]
    for i in range(n_s):
        b = make_babble(tones, i, fs, np.random.default_rng(i))
        spec = np.abs(np.fft.rfft(b)) ** 2
        power_at = {j: spec[100 * (j + 1)] for j in range(n_s)}
        present = [j for j in range(n_s) if power_at[j] > 1e-6 * spec.max()]
        assert i not in present
        assert len(present) == config.N_BABBLE_TALKERS


def test_babble_unit_rms_and_seeded():
    pool = [s for _, s in synthetic_pool(8)]
    a = make_babble(pool, 0, 12345, np.random.default_rng(3))
    b = make_babble(pool, 0, 12345, np.random.default_rng(3))
    assert a.shape == (12345,) and np.all(np.isfinite(a))
    assert np.sqrt(np.mean(a**2)) == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_array_equal(a, b)


def test_extract_clean_zip_uses_base_names_only(tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    wav = io.BytesIO()
    sf.write(wav, np.zeros(800), config.FS, format="WAV", subtype="PCM_16")
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("clean/sp01.wav", wav.getvalue())
        z.writestr("../escape.wav", wav.getvalue())  # must not leave the target folder
        z.writestr("clean/readme.txt", "ignored")
    out = tmp_path / "clean"
    written = extract_clean_zip(io.BytesIO(buf.getvalue()), out)
    assert [p.name for p in written] == ["escape.wav", "sp01.wav"]
    assert all(p.parent == out for p in written)
    assert not (tmp_path / "escape.wav").exists()
    assert len(load_noizeus(out)) == 2


# --- python main.py --input: loading any recording -----------------------------
def test_load_audio_file_mixes_stereo_down_and_resamples(tmp_path):
    fs_in = 48000
    t = np.arange(fs_in) / fs_in
    tone = np.sin(2 * np.pi * 300 * t)
    sf.write(tmp_path / "st.wav", np.column_stack([0.5 * tone, 0.1 * tone]), fs_in,
             subtype="FLOAT")
    x = load_audio_file(tmp_path / "st.wav")
    assert x.shape == (config.FS,) and x.dtype == np.float64
    t8 = np.arange(config.FS) / config.FS
    mid = slice(500, config.FS - 500)  # away from the resampling filter's edges
    np.testing.assert_allclose(x[mid], 0.3 * np.sin(2 * np.pi * 300 * t8)[mid], atol=1e-3)


def test_load_audio_file_keeps_8k_mono_unchanged(tmp_path):
    y = np.random.default_rng(0).uniform(-0.5, 0.5, 4000)
    sf.write(tmp_path / "m.wav", y, config.FS, subtype="DOUBLE")
    np.testing.assert_array_equal(load_audio_file(tmp_path / "m.wav"), y)


def test_load_audio_file_rejects_empty(tmp_path):
    sf.write(tmp_path / "e.wav", np.zeros(0), config.FS)
    with pytest.raises(ValueError, match="no samples"):
        load_audio_file(tmp_path / "e.wav")
