import csv
import tempfile
import unittest
from pathlib import Path

from pii_prep.taxonomy import load_taxonomy


class TaxonomyTests(unittest.TestCase):
    def test_csv_names_map_to_training_labels_without_raw_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "taxonomy.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Name", "Note", "Data Classification", "Category Type"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Name": "TFN",
                        "Note": "tax file number",
                        "Data Classification": "Highly Protected",
                        "Category Type": "Government",
                    }
                )
                writer.writerow(
                    {
                        "Name": "Passport Number",
                        "Note": "passport",
                        "Data Classification": "Highly Protected",
                        "Category Type": "Government",
                    }
                )

            taxonomy = load_taxonomy(csv_path)

        self.assertIn("AU_TFN", taxonomy.labels)
        self.assertIn("PASSPORT_NUMBER", taxonomy.labels)
        self.assertIn("NON_PII", taxonomy.labels)
        self.assertNotIn("AU_PASSPORT", taxonomy.labels)
        self.assertEqual(taxonomy.classification_for("PASSPORT_NUMBER"), "Highly Protected")


if __name__ == "__main__":
    unittest.main()
