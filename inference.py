# inference.py
# Module inference untuk klasifikasi senyawa inhibitor BACE-1.
# Mendukung 4 varian model:
#   - GIN  + scaffold split
#   - GIN  + stratified split
#   - GINE + scaffold split
#   - GINE + stratified split
#
# Setiap varian adalah ensemble 5 model dari multi-seed training.
#
# Class utama: MultiModelPredictor
# Method utama:
#   - predict(smiles, model_key)             -> prediksi 1 varian
#   - predict_all(smiles)                    -> prediksi semua varian
#   - predict_batch(smiles_list, model_key)  -> batch prediksi 1 varian

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Batch

from model_def import create_model
from molecular_utils import (
    preprocess_smiles,
    compute_properties,
    check_out_of_domain,
)


# ============================================================
# Konfigurasi
# ============================================================

ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]

CLASSIFICATION_THRESHOLD = 0.5

HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.65
HIGH_UNCERTAINTY_THRESHOLD = 0.10


# ============================================================
# Daftar model varian yang didukung
# ============================================================

# Setiap entry: (model_type, split, prefix_file, subfolder)
# prefix_file digunakan untuk nama file gin_seed_*.pt atau gine_seed_*.pt
MODEL_VARIANTS = {
    'gin_scaffold': {
        'model_type': 'GIN',
        'split':      'scaffold',
        'prefix':     'gin',
        'subfolder':  os.path.join('gin', 'scaffold'),
        'label':      'GIN · scaffold',
        'description': 'GIN tanpa edge features, scaffold split (uji generalisasi scaffold baru)',
    },
    'gin_stratified': {
        'model_type': 'GIN',
        'split':      'stratified',
        'prefix':     'gin',
        'subfolder':  os.path.join('gin', 'stratified'),
        'label':      'GIN · stratified',
        'description': 'GIN tanpa edge features, stratified split (uji performa i.i.d.)',
    },
    'gine_scaffold': {
        'model_type': 'GINE',
        'split':      'scaffold',
        'prefix':     'gine',
        'subfolder':  os.path.join('gine', 'scaffold'),
        'label':      'GINE · scaffold',
        'description': 'GINE dengan edge features, scaffold split',
    },
    'gine_stratified': {
        'model_type': 'GINE',
        'split':      'stratified',
        'prefix':     'gine',
        'subfolder':  os.path.join('gine', 'stratified'),
        'label':      'GINE · stratified',
        'description': 'GINE dengan edge features, stratified split (default, PR-AUC tertinggi)',
    },
}

DEFAULT_MODEL_KEY = 'gine_stratified'


# ============================================================
# Class utama: MultiModelPredictor
# ============================================================

class MultiModelPredictor:
    # Multi-model predictor untuk klasifikasi inhibitor BACE-1.
    # Load 4 varian (GIN/GINE x scaffold/stratified), masing-masing
    # adalah ensemble 5 seeds.

    def __init__(self, models_dir='models', device=None, variants=None):
        # Args:
        #   models_dir: Path ke folder root models/
        #   device:     'cpu', 'cuda', atau None (auto)
        #   variants:   list of variant keys yang mau di-load.
        #               None = load semua yang tersedia.

        self.models_dir = models_dir

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        if variants is None:
            variants = list(MODEL_VARIANTS.keys())

        # Storage: variant_key -> {'models': [...], 'config': {...}, 'meta': {...}}
        self.variants = {}
        self.variant_metadata = {}

        for vkey in variants:
            self._load_variant(vkey)

        if len(self.variants) == 0:
            raise RuntimeError(
                f'Tidak ada varian model yang berhasil di-load dari {models_dir}. '
                f'Pastikan struktur folder benar (lihat README).'
            )

        print(f'Loaded {len(self.variants)} varian: {list(self.variants.keys())}')

    def _load_variant(self, variant_key):
        # Load 5 seed model untuk satu varian.
        if variant_key not in MODEL_VARIANTS:
            print(f'Warning: variant {variant_key} tidak dikenal, skip.')
            return

        vinfo = MODEL_VARIANTS[variant_key]
        variant_dir = os.path.join(self.models_dir, vinfo['subfolder'])

        if not os.path.isdir(variant_dir):
            print(f'Warning: folder {variant_dir} tidak ada, skip varian {variant_key}.')
            return

        # 1. Coba load best_model.pt untuk dapatkan config + metadata
        best_path = os.path.join(variant_dir, f"{vinfo['prefix']}_best_model.pt")
        if not os.path.exists(best_path):
            print(f'Warning: {best_path} tidak ada, skip varian {variant_key}.')
            return

        try:
            ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'Warning: gagal load {best_path}: {e}')
            return

        config = ckpt.get('config', None)
        if config is None:
            print(f'Warning: {best_path} tidak punya config, skip varian {variant_key}.')
            return

        # 2. Load 5 seed model
        models = []
        seeds_loaded = []
        for seed in ENSEMBLE_SEEDS:
            seed_path = os.path.join(variant_dir, f"{vinfo['prefix']}_seed_{seed}.pt")
            if not os.path.exists(seed_path):
                print(f'Warning: {seed_path} tidak ada, skip seed.')
                continue

            try:
                model = create_model(vinfo['model_type'], config)
                state_dict = torch.load(seed_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                models.append(model)
                seeds_loaded.append(seed)
            except Exception as e:
                print(f'Warning: gagal load {seed_path}: {e}')
                continue

        if len(models) == 0:
            print(f'Warning: tidak ada seed model untuk varian {variant_key}, skip.')
            return

        self.variants[variant_key] = {
            'models':       models,
            'seeds_loaded': seeds_loaded,
            'config':       config,
            'model_type':   vinfo['model_type'],
            'split':        vinfo['split'],
            'uses_edge':    vinfo['model_type'] == 'GINE',
        }

        # Simpan metadata untuk ditampilkan di sidebar UI
        self.variant_metadata[variant_key] = {
            'label':            vinfo['label'],
            'description':      vinfo['description'],
            'model_type':       vinfo['model_type'],
            'split':            vinfo['split'],
            'config':           config,
            'best_seed':        ckpt.get('best_seed', None),
            'test_metrics':     ckpt.get('test_metrics', {}),
            'multi_seed_mean':  ckpt.get('multi_seed_mean', {}),
            'multi_seed_std':   ckpt.get('multi_seed_std', {}),
            'n_seeds_loaded':   len(seeds_loaded),
        }

        print(f'  {variant_key}: loaded {len(models)} seeds, '
              f"arch={vinfo['model_type']} L={config['num_layers']} "
              f"H={config['hidden_dim']} dropout={config['dropout']}")

    @torch.no_grad()
    def _ensemble_predict(self, graph, variant_key):
        # Inference dengan ensemble 5 model untuk satu varian.
        # Returns: (probs_per_model, mean_prob, std_prob)
        variant = self.variants[variant_key]
        graph_dev = graph.to(self.device)
        batch = Batch.from_data_list([graph_dev])

        probs = []
        for model in variant['models']:
            out = model(batch)
            prob = F.softmax(out, dim=1)[:, 1].item()
            probs.append(prob)

        probs = np.array(probs)
        return probs, float(probs.mean()), float(probs.std())

    def predict(self, smiles, model_key=DEFAULT_MODEL_KEY):
        # Prediksi single SMILES dengan satu varian model.

        if model_key not in self.variants:
            return {
                'input_smiles': smiles,
                'success':      False,
                'error':        f'Variant {model_key} tidak ter-load',
                'model_key':    model_key,
            }

        result = {
            'input_smiles':     smiles,
            'model_key':        model_key,
            'success':          False,
            'error':            None,
            'canonical_smiles': None,
            'classification':   None,
            'probability':      None,
            'uncertainty':      None,
            'confidence_level': None,
            'probs_per_model':  None,
            'properties':       None,
            'in_domain':        None,
            'domain_warnings':  [],
        }

        # Step 1: preprocessing SMILES
        prep = preprocess_smiles(smiles)
        if not prep['success']:
            result['error'] = prep['error']
            return result

        canonical = prep['canonical_smiles']
        graph = prep['graph']
        result['canonical_smiles'] = canonical

        # Step 2: properti molekul
        result['properties'] = compute_properties(canonical)

        # Step 3: cek out-of-domain
        in_domain, warnings = check_out_of_domain(canonical)
        result['in_domain'] = in_domain
        result['domain_warnings'] = warnings

        # Step 4: ensemble prediction
        try:
            probs_per_model, mean_prob, std_prob = self._ensemble_predict(graph, model_key)
        except Exception as e:
            result['error'] = f'Inference error: {str(e)}'
            return result

        # Step 5: format hasil
        is_active = mean_prob >= CLASSIFICATION_THRESHOLD
        confidence_level = self._get_confidence_level(mean_prob, std_prob)

        result['success'] = True
        result['classification'] = 'Aktif' if is_active else 'Tidak Aktif'
        result['probability'] = mean_prob
        result['uncertainty'] = std_prob
        result['confidence_level'] = confidence_level
        result['probs_per_model'] = probs_per_model.tolist()
        result['seeds_used'] = self.variants[model_key]['seeds_loaded']

        return result

    def predict_all(self, smiles):
        # Prediksi single SMILES dengan SEMUA varian yang ter-load.
        # Berguna untuk mode "Compare 4 models".
        # Returns: dict {variant_key: result_dict}

        results = {}
        for vkey in self.variants.keys():
            results[vkey] = self.predict(smiles, model_key=vkey)
        return results

    def predict_batch(self, smiles_list, model_key=DEFAULT_MODEL_KEY,
                       show_progress=False):
        # Batch prediksi dengan satu varian.

        results = []
        total = len(smiles_list)
        for idx, smi in enumerate(smiles_list):
            if show_progress and (idx + 1) % 10 == 0:
                print(f'  Processed: {idx + 1}/{total}')
            results.append(self.predict(smi, model_key=model_key))

        # Flatten ke DataFrame
        rows = []
        for idx, res in enumerate(results):
            row = {
                'index':            idx,
                'input_smiles':     res['input_smiles'],
                'success':          res['success'],
                'error':            res['error'],
                'canonical_smiles': res.get('canonical_smiles'),
                'classification':   res.get('classification'),
                'probability':      res.get('probability'),
                'uncertainty':      res.get('uncertainty'),
                'confidence_level': res.get('confidence_level'),
                'in_domain':        res.get('in_domain'),
            }
            props = res.get('properties')
            if props:
                row['MW']        = props['MW']
                row['AlogP']     = props['AlogP']
                row['num_atoms'] = props['num_atoms']
                row['num_bonds'] = props['num_bonds']
                row['num_rings'] = props['num_rings']
            else:
                row['MW']        = None
                row['AlogP']     = None
                row['num_atoms'] = None
                row['num_bonds'] = None
                row['num_rings'] = None

            warnings = res.get('domain_warnings', [])
            row['domain_warnings'] = '; '.join(warnings) if warnings else ''
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _get_confidence_level(mean_prob, std_prob):
        # Tentukan confidence level (High/Medium/Low).
        prob_distance = abs(mean_prob - 0.5)
        if prob_distance >= (HIGH_CONFIDENCE_THRESHOLD - 0.5) and std_prob < HIGH_UNCERTAINTY_THRESHOLD:
            return 'High'
        if prob_distance < (LOW_CONFIDENCE_THRESHOLD - 0.5) or std_prob >= HIGH_UNCERTAINTY_THRESHOLD:
            return 'Low'
        return 'Medium'

    def get_loaded_variants(self):
        # Return list variant keys yang berhasil di-load.
        return list(self.variants.keys())

    def get_variant_info(self, variant_key):
        # Return metadata varian (label, config, test metrics).
        return self.variant_metadata.get(variant_key, None)

    def get_default_model_key(self):
        # Return default model key. Fallback ke first loaded kalau default tidak ada.
        if DEFAULT_MODEL_KEY in self.variants:
            return DEFAULT_MODEL_KEY
        loaded = self.get_loaded_variants()
        return loaded[0] if loaded else None


# ============================================================
# Test ringan saat dijalankan langsung
# ============================================================

if __name__ == '__main__':
    print('Loading MultiModelPredictor...')
    predictor = MultiModelPredictor(models_dir='models')

    print('\nLoaded variants:')
    for vkey in predictor.get_loaded_variants():
        info = predictor.get_variant_info(vkey)
        print(f'  {vkey}: {info["label"]} '
              f'(arch={info["model_type"]}, split={info["split"]}, '
              f'seeds={info["n_seeds_loaded"]})')

    test_smiles = [
        'O=S1(=O)CC(C)(c2ccc(NC(=O)c3cc(F)cnc3C)cc2F)N=C(N)N1C',  # Verubecestat
        'CCO',                                                       # OOD
        'invalid_xyz',                                               # Invalid
    ]

    default_key = predictor.get_default_model_key()
    print(f'\nDefault model: {default_key}')
    print('\nTest predictions (default model only):')
    for smi in test_smiles:
        print(f'\nSMILES: {smi[:60]}')
        result = predictor.predict(smi)
        if result['success']:
            print(f'  Classification : {result["classification"]}')
            print(f'  P(active) μ    : {result["probability"]:.4f}')
            print(f'  Uncertainty σ  : {result["uncertainty"]:.4f}')
            print(f'  Confidence     : {result["confidence_level"]}')
            print(f'  In domain      : {result["in_domain"]}')
        else:
            print(f'  ERROR: {result["error"]}')

    print('\nTest compare mode (all 4 variants):')
    compare = predictor.predict_all(test_smiles[0])
    for vkey, res in compare.items():
        if res['success']:
            print(f'  {vkey:20s} -> {res["classification"]:12s} '
                  f'P={res["probability"]:.4f} σ={res["uncertainty"]:.4f}')
        else:
            print(f'  {vkey:20s} -> ERROR: {res["error"]}')

    print('\nTest selesai.')