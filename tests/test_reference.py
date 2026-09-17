"""CPU-only tests: independent recurrence, shared inputs and validation."""
import dataclasses
from pathlib import Path
import numpy as np
import pytest
from m5diffusion.scheduler.dpm import schedule, initial_noise
from m5diffusion.engine.common import Request, inputs, ROOT

CONFIG = {'beta_start': .00085, 'beta_end': .012, 'num_train_timesteps': 1000}

@pytest.mark.parametrize('steps', [2, 3, 20, 100])
def test_coefficients_match_independent_dpmpp_midpoint_recurrence(steps):
    sigmas, _, coeff = schedule(steps, CONFIG)
    # Analytic denoiser deliberately varies with x and sigma; an old-state/sign
    # error therefore cannot pass simply because all denoised values coincide.
    initial = np.random.default_rng(71).normal(size=(2, 4, 3))
    actual, expected = initial.copy(), initial.copy()
    old_actual = np.zeros_like(actual)
    old_expected = None
    previous_time = None
    for i, (sigma, ratio, factor, ca, cb) in enumerate(coeff):
        den_actual = np.tanh(actual / (1 + float(sigma))) + .01 * i
        actual = ratio * actual + factor * (ca * den_actual + cb * old_actual)
        old_actual = den_actual
        den = np.tanh(expected / (1 + float(sigmas[i]))) + .01 * i
        now = -np.log(float(sigmas[i]))
        if sigmas[i + 1] == 0:
            expected = den
        else:
            next_time = -np.log(float(sigmas[i + 1]))
            dt = next_time - now
            corrected = den if old_expected is None else den + (den - old_expected) * dt / (2 * (now - previous_time))
            expected = np.exp(-dt) * expected - np.expm1(-dt) * corrected
        old_expected, previous_time = den, now
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

@pytest.mark.parametrize('changes', [dict(width=63), dict(height=1088), dict(steps=1), dict(steps=101), dict(cfg=float('nan')), dict(cfg=float('inf')), dict(cfg=.5)])
def test_request_rejects_out_of_range(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(Request(), **changes).validate()

@pytest.mark.parametrize('width,height', [(64, 64), (512, 768), (1024, 1024)])
def test_noise_shape_layout_and_seed(width, height):
    noise = initial_noise(42, width, height)
    assert noise.shape == (1, 4, height // 8, width // 8)
    assert noise.dtype == np.float32 and np.isfinite(noise).all()
    np.testing.assert_array_equal(noise, initial_noise(42, width, height))
    np.testing.assert_array_equal(noise, noise.transpose(0, 2, 3, 1).transpose(0, 3, 1, 2))
    assert not np.array_equal(noise, initial_noise(43, width, height))

def test_real_tokenizer_padding_truncation_and_cfg_order():
    model = ROOT / 'models/sd15'
    if not (model / 'tokenizer/tokenizer_config.json').exists():
        pytest.skip('local SD1.5 tokenizer unavailable; no network downloads in CPU tests')
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(str(model / 'tokenizer'), local_files_only=True)
    r = Request(prompt='a red house ' * 100, negative_prompt='fog', width=512, height=768)
    ids, noise, sigmas, ts, coeff = inputs(model, r)
    assert ids.shape == (2, 77) and np.issubdtype(ids.dtype, np.integer)
    for i, text in enumerate([r.negative_prompt, r.prompt]):
        reference = tokenizer(text, padding='max_length', max_length=77, truncation=True, return_tensors='np').input_ids[0]
        np.testing.assert_array_equal(ids[i], reference)
    assert noise.shape == (1, 4, 96, 64)
    assert sigmas.shape == (21,) and ts.shape == (20,) and coeff.shape == (20, 5)
    assert ids[1, -1] == tokenizer.eos_token_id

@pytest.mark.parametrize('changes', [dict(width=512.0), dict(steps=2.5), dict(seed=-1), dict(seed=True), dict(prompt=None), dict(cfg='7'), dict(cfg=True)])
def test_request_rejects_invalid_types_and_seed(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(Request(), **changes).validate()


def test_summary_does_not_compare_mismatched_requests(tmp_path):
    import json
    from summarize import build_report
    common = {'request': {'seed': 42}, 'model': 'same', 'sampler': 'DPM++ 2M', 'schedule': 'Karras', 'rng': 'PCG64', 'precision': 'same', 'runs': [{'cold': True, 'total_time': 50}, {'cold': False, 'total_time': 10}, {'cold': False, 'total_time': 20}]}
    (tmp_path / 'base.json').write_text(json.dumps(common))
    candidate = {**common, 'runs': [{'cold': False, 'total_time': 5}]}
    (tmp_path / 'candidate.json').write_text(json.dumps(candidate))
    mismatch = {**candidate, 'request': {'seed': 43}}
    (tmp_path / 'mismatch.json').write_text(json.dumps(mismatch))
    result = build_report(tmp_path, 'base.json')
    matched = next(r for r in result['comparisons'] if r['candidate'] == 'candidate.json')
    assert matched['total_time_warm_speedup'] == 3
    unmatched = next(r for r in result['comparisons'] if r['candidate'] == 'mismatch.json')
    assert unmatched['protocol_mismatches'] == ['request']
    assert 'total_time_warm_speedup' not in unmatched
    assert next(r for r in result['runs'] if r['file'] == 'base.json')['cold']['stages']['total_time']['median'] == 50


def test_summary_never_divides_legacy_stage_sum_by_full_wall(tmp_path):
    import json
    from summarize import build_report
    old = {'request': {}, 'runs': [{'cold': False, 'total_time': 10, 'diffusion_time': 8}]}
    new = {'request': {}, 'timing_schema': 'generation-wall-v2', 'runs': [{'cold': False, 'total_time': 6, 'wall_time': 6, 'diffusion_time': 4, 'stage_sum_time': 5.9}]}
    (tmp_path / 'old.json').write_text(json.dumps(old))
    (tmp_path / 'new.json').write_text(json.dumps(new))
    report = build_report(tmp_path, 'old.json')
    comparison = report['comparisons'][0]
    assert comparison['timing_schema_match'] is False
    assert 'total_time_warm_speedup' not in comparison
    assert comparison['diffusion_time_warm_speedup'] == 2
    (tmp_path / 'new-baseline.json').write_text(json.dumps(new))
    matched = next(c for c in build_report(tmp_path, 'new-baseline.json')['comparisons'] if c['candidate'] == 'new.json')
    assert matched['total_time_warm_speedup'] == 1
