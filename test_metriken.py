import unittest
from metrics import berechne_metriken

class TestMetriken(unittest.TestCase):

    def test_perfect_match(self):
        """Testet, ob ein identischer Abgleich 100% (1.0) bei allen Metriken ergibt."""
        gt = {"Medikament": ["Aspirin"], "Dosierung": ["500 mg"]}
        pred = {"Medikament": ["Aspirin"], "Dosierung": ["500 mg"]}
        
        p, r, f1, tp, fp, fn = berechne_metriken(gt, pred)
        
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f1, 1.0)

    def test_ignored_category(self):
        """Testet, ob die Kategorie 'Krankheit' wie gefordert komplett ignoriert wird."""
        gt = {"Krankheit": ["Asthma"], "Medikament": ["Salbutamol"]}
        # Das Modell halluziniert "Pneumonie" als Krankheit dazu
        pred = {"Krankheit": ["Pneumonie", "Asthma"], "Medikament": ["Salbutamol"]}
        
        p, r, f1, tp, fp, fn = berechne_metriken(gt, pred)
        
        # TP sollte 1 sein (nur Salbutamol), FP sollte 0 sein (Pneumonie wird ignoriert)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_case_insensitivity_and_strip(self):
        """Testet, ob Groß-/Kleinschreibung und überflüssige Leerzeichen ignoriert werden."""
        gt = {"Medikament": ["  Ibuprofen "]}
        pred = {"Medikament": ["ibuprofen"]}
        
        _, _, _, tp, _, _ = berechne_metriken(gt, pred)
        self.assertEqual(tp, 1)  # Sollte als identisch erkannt werden

    def test_false_positives_and_negatives(self):
        """Testet die korrekte Berechnung von FP und FN (Precision und Recall = 0.5)."""
        gt = {"Medikament": ["Aspirin", "Ibuprofen"]}
        pred = {"Medikament": ["Aspirin", "Paracetamol"]}
        
        p, r, f1, tp, fp, fn = berechne_metriken(gt, pred)
        
        self.assertEqual(tp, 1)  # Aspirin
        self.assertEqual(fp, 1)  # Paracetamol (zu viel)
        self.assertEqual(fn, 1)  # Ibuprofen (vergessen)
        self.assertEqual(p, 0.5)
        self.assertEqual(r, 0.5)
        self.assertEqual(f1, 0.5)

    def test_empty_inputs(self):
        """Testet die Division-by-Zero-Absicherung bei leeren Eingaben."""
        p, r, f1, tp, fp, fn = berechne_metriken({}, {})
        
        self.assertEqual(tp, 0)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

if __name__ == '__main__':
    unittest.main()