import unittest
from unittest.mock import patch

from app import haystack_rag


class FakeRetrievalPipeline:
    def __init__(self, documents):
        self._documents = documents

    def run(self, payload):
        self.payload = payload
        return {"retriever": {"documents": self._documents}}


class HaystackRagRetrievalFilterTests(unittest.TestCase):
    def test_build_retrieval_filters_combines_domain_and_intention_metadata(self):
        with patch.object(
            haystack_rag,
            "_get_intention_label_from_code",
            return_value="Structurer un reporting fiable",
        ):
            filters = haystack_rag._build_retrieval_filters(
                domaine_code="finance_pilotage",
                intention_code="2",
            )

        self.assertIsNotNone(filters)
        self.assertEqual(filters["operator"], "AND")
        self.assertEqual(len(filters["conditions"]), 2)

        domain_filter, intention_filter = filters["conditions"]
        self.assertEqual(domain_filter["operator"], "OR")
        self.assertEqual(intention_filter["operator"], "OR")

        domain_values = {condition["value"] for condition in domain_filter["conditions"]}
        self.assertIn("finance_pilotage", domain_values)
        self.assertIn("Finances & rentabilité", domain_values)

        intention_values = {condition["value"] for condition in intention_filter["conditions"]}
        self.assertEqual(intention_values, {"Structurer un reporting fiable"})

    def test_retrieve_docs_for_question_applies_metadata_filters_before_retrieval(self):
        captured = {}
        fake_pipeline = FakeRetrievalPipeline(documents=[["doc-a", "doc-b"]])

        def fake_build_pipeline(filters=None):
            captured["filters"] = filters
            return fake_pipeline

        with (
            patch.object(haystack_rag, "_get_intention_label_from_code", return_value="Structurer un reporting fiable"),
            patch.object(haystack_rag, "build_rag_retrieval_only_pipeline", side_effect=fake_build_pipeline),
        ):
            docs = haystack_rag._retrieve_docs_for_question(
                "rapport solide et comprehensible",
                selected_domain_code="finance_pilotage",
                selected_intention="2",
            )

        self.assertEqual(docs, ["doc-a", "doc-b"])
        self.assertEqual(fake_pipeline.payload, {"embedder": {"text": "rapport solide et comprehensible"}})

        filters = captured["filters"]
        self.assertIsNotNone(filters)
        self.assertEqual(filters["operator"], "AND")

        domain_filter, intention_filter = filters["conditions"]
        domain_values = {condition["value"] for condition in domain_filter["conditions"]}
        intention_values = {condition["value"] for condition in intention_filter["conditions"]}

        self.assertIn("finance_pilotage", domain_values)
        self.assertIn("Finances & rentabilité", domain_values)
        self.assertEqual(intention_values, {"Structurer un reporting fiable"})


class HaystackRagCaseExtraFieldsTests(unittest.TestCase):
    def test_doc_to_case_dict_reads_canonical_and_alias_headers(self):
        class _Doc:
            id = "row-1"
            content = "Texte RAG"
            meta = {
                "effort": "Moyen",
                "prerequis_donnees": "Export CSV",
                "Guardrails": "Vérifier sources",
                "questions_qualification": "Q1 ?\nQ2 ?",
                "data_sensitivity": "Données perso",
            }

        row = haystack_rag._doc_to_case_dict(_Doc(), 0)
        self.assertEqual(row["id"], "row-1")
        self.assertEqual(row["content"], "Texte RAG")
        self.assertEqual(row["effort"], "Moyen")
        self.assertEqual(row["prerequis_donnees"], "Export CSV")
        self.assertEqual(row["guardrails"], "Vérifier sources")
        self.assertEqual(row["questions_qualification"], "Q1 ?\nQ2 ?")
        self.assertEqual(row["sensibilite_donnees"], "Données perso")

    def test_docs_to_payload_aligns_extras_with_contents(self):
        class _Doc:
            def __init__(self, mid, text, meta):
                self.id = mid
                self.content = text
                self.meta = meta

        docs = [
            _Doc("a", "c1", {"effort": "Faible"}),
            _Doc("b", "c2", {}),
        ]
        _s, ids, contents, extras = haystack_rag._docs_to_payload(docs)
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(contents, ["c1", "c2"])
        self.assertEqual(extras[0]["effort"], "Faible")
        self.assertIsNone(extras[1]["effort"])


if __name__ == "__main__":
    unittest.main()
