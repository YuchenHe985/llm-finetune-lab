import json
import os
import sqlite3
import tempfile
import unittest

import abstain_eval
import sql_exec_eval as eval_sql


EXAMPLE = {
    "context": "CREATE TABLE city (name TEXT, population INTEGER);",
    "question": "Which city has population above 10?",
    "answer": "SELECT name FROM city WHERE population > 10",
}


class SqlExecutionTests(unittest.TestCase):
    def test_first_statement_and_normalisation(self):
        self.assertEqual(eval_sql.first_statement("```sql\nSELECT 1;\n```"), "SELECT 1")
        self.assertEqual(eval_sql.normalise(" SELECT  1; "), "select 1")

    def test_exact_and_invalid_predictions(self):
        self.assertEqual(eval_sql.classify(EXAMPLE, EXAMPLE["answer"])[0], "exact")
        self.assertEqual(eval_sql.classify(EXAMPLE, "population > 10")[0], "error")

    def test_query_budget_interrupts_runaway_statement(self):
        con = sqlite3.connect(":memory:")
        with self.assertRaises(sqlite3.OperationalError):
            eval_sql.run(con, "WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t) SELECT * FROM t")

    def test_jsonl_alignment_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            eval_sql.require_aligned([1, 2], [1])
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(json.dumps({"a": 1}) + "\n\n")
            path = f.name
        try:
            self.assertEqual(eval_sql.read_jsonl(path), [{"a": 1}])
        finally:
            os.unlink(path)

    def test_explicit_limit_scores_a_matching_prefix(self):
        examples, predictions = eval_sql.apply_limit(list(range(10)), list(range(4)), 4)
        eval_sql.require_aligned(examples, predictions)
        self.assertEqual(examples, [0, 1, 2, 3])
        self.assertEqual(predictions, [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            eval_sql.apply_limit([], [], -1)

    def test_invented_constant_signal(self):
        self.assertFalse(abstain_eval.invented_constant(EXAMPLE, "SELECT name FROM city WHERE name = 'city'"))
        self.assertTrue(abstain_eval.invented_constant(EXAMPLE, "SELECT name FROM city WHERE name = 'Boston'"))


if __name__ == "__main__":
    unittest.main()
