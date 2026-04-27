import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_full_4b


class TrainFull4bTests(unittest.TestCase):
    def test_default_paths_target_qwen35_4b_base_and_processed_json_spans(self):
        cfg = train_full_4b.build_run_config(profile="smoke")

        self.assertEqual(cfg.base_model, "/home/admin/model/Qwen3.5-4B-Base")
        self.assertEqual(cfg.train_path, "../data/processed/qwen_sft_train.jsonl")
        self.assertEqual(cfg.dev_path, "../data/processed/qwen_sft_dev.jsonl")
        self.assertEqual(cfg.meta_path, "../data/processed/meta.json")
        self.assertEqual(cfg.output_dir, "../outputs/qwen3_5_4b_base_full_73class")

    def test_profiles_are_full_parameter_not_lora(self):
        cfg = train_full_4b.build_run_config(profile="safe_full")

        self.assertTrue(cfg.full_finetune)
        self.assertIsNone(cfg.peft_config)
        self.assertEqual(cfg.learning_rate, 2e-5)
        self.assertEqual(cfg.optim, "paged_adamw_8bit")

    def test_smoke_profile_limits_training_steps(self):
        cfg = train_full_4b.build_run_config(profile="smoke")

        self.assertEqual(cfg.max_steps, 20)
        self.assertEqual(cfg.num_train_epochs, 1)
        self.assertEqual(cfg.per_device_train_batch_size, 1)

    def test_unknown_profile_fails_fast(self):
        with self.assertRaises(ValueError):
            train_full_4b.build_run_config(profile="bad-profile")


if __name__ == "__main__":
    unittest.main()
