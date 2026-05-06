import sys, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/home/admin/ZYX/pii_training_prep_v3_2/src')

from pii_prep.qwen_spancls_cache import main

print('Building Qwen4B span embedding cache (writes directly to qwen4b_spancls_embeddings_*)...')
main(argv=[
    '--model-path', '/home/admin/model/Qwen3.5-4B-Base',
    '--batch-size', '8',
    '--splits', 'train,dev,test',
    '--cache-name-prefix', 'qwen4b_spancls_embeddings',
])
