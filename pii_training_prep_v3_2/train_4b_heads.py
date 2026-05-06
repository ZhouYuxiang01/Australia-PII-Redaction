import sys, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/home/admin/ZYX/pii_training_prep_v3_2/src')

from pii_prep.qwen_spancls_heads import main

print('Training Qwen4B span heads on qwen4b_spancls_embeddings cache...')
main(argv=[
    '--cache-name-prefix', 'qwen4b_spancls_embeddings',
    '--run-dir-name', 'qwen4b_spancls_heads',
    '--report-prefix', 'stage3a_qwen4b_head',
])
