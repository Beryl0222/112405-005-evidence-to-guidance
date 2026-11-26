"""evidence_workbench 的规则与端到端测试。

运行：
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_workbench import (  # noqa: E402
    audit_wording,
    diff_reports,
    impacted_guidance_ids,
    process_case,
    review_after_change,
)
from evidence_workbench.__main__ import main as cli_main  # noqa: E402
from evidence_workbench.nlp import extract_claims_from_text, parse_guidance_text  # noqa: E402

CASE_PATH = ROOT / "data" / "case.json"
CASE = json.loads(CASE_PATH.read_text(encoding="utf-8"))


def base_report():
    return process_case(CASE)


def review_by_id(report, gid):
    return next(r for r in report["guidance_reviews"] if r["guidance_id"] == gid)


def check_codes(review):
    return {c["code"] for c in review["checks"]}


class ExtractionAB7Test(unittest.TestCase):
    def setUp(self):
        self.text = next(r["text"] for r in CASE["records"] if r["id"] == "AB-7")
        self.claims = extract_claims_from_text(self.text)

    def test_two_propositions_extracted(self):
        self.assertEqual(len(self.claims), 2)

    def test_surrogate_claim_fields(self):
        c = next(c for c in self.claims if c["claim_type"] == "result")
        self.assertEqual(c["intervention"], "甲方案")
        self.assertEqual(c["outcome"], "替代指标")
        self.assertEqual(c["outcome_class"], "surrogate")
        self.assertEqual(c["direction"], "favorable")
        self.assertEqual(c["evidential_force"], "supports_benefit_claim")
        self.assertIn("60岁", c["population"])
        self.assertIn("基线指标偏高", c["population"])
        kinds = {(q["type"], q.get("min_age")) for q in c["population_qualifiers"]}
        self.assertIn(("min_age", 60), kinds)
        self.assertTrue(any(q["type"] == "eligibility_condition" for q in c["population_qualifiers"]))

    def test_unmeasured_hospitalization_is_gap_not_negative(self):
        c = next(c for c in self.claims if c["claim_type"] == "outcome_not_measured")
        self.assertEqual(c["outcome_concept"], "hospitalization")
        self.assertEqual(c["outcome_class"], "patient_important")
        self.assertEqual(c["direction"], "not_assessed")
        self.assertEqual(c["evidential_force"], "outcome_not_studied")

    def test_source_sentences_are_exact_substrings(self):
        for c in self.claims:
            self.assertIn(c["source_sentence"], self.text)
            self.assertIn(c["source_clause"], self.text)
            self.assertTrue(c["source_sentence"].strip())


class GuidanceParsingTest(unittest.TestCase):
    def test_g3_parse(self):
        g = parse_guidance_text("高龄人群可考虑甲方案降低住院风险。")
        self.assertEqual(g["intervention"], "甲方案")
        self.assertEqual(g["outcome_concept"], "hospitalization")
        self.assertEqual(g["desired_direction"], "decrease")
        self.assertIn("高龄", g["population"])


class NullResultSemanticsTest(unittest.TestCase):
    def _one(self, text):
        cs = extract_claims_from_text(text)
        self.assertTrue(cs, f"未抽出主张：{text}")
        return cs[0]

    def test_common_null_phrasings_are_inconclusive(self):
        for text in [
            "两组住院率差异无统计学意义。",
            "研究未发现心血管事件显著减少。",
            "新证据不足，尚不能证实甲方案降低死亡风险。",
            "与安慰剂相比，认知评分未见明显改善。",
        ]:
            c = self._one(text)
            self.assertEqual(c["direction"], "no_significant_difference", text)
            self.assertEqual(c["evidential_force"], "inconclusive_no_detected_difference", text)

    def test_negated_verb_is_not_positive(self):
        c = self._one("试验组与对照组相比，住院率并未降低。")
        self.assertEqual(c["direction"], "no_significant_difference")

    def test_null_is_never_labeled_no_effect(self):
        for text in ["两组差异无统计学意义。", "未发现显著差异。"]:
            c = self._one(text)
            self.assertNotEqual(c["direction"], "no_effect_demonstrated")

    def test_comparator_extracted(self):
        c = self._one("丙方案与常规治疗相比使心肌梗死发生率下降。")
        self.assertIn("常规治疗", c["comparator"])
        c2 = self._one("戊补充剂对认知评分的影响与安慰剂相比未发现显著差异。")
        self.assertEqual(c2["comparator"], "安慰剂")


class WordingAuditTest(unittest.TestCase):
    def test_flags_causal_overclaim(self):
        issues = audit_wording("新研究未发现两组住院率有差异，因此甲方案被证明无效。", source_id="X")
        self.assertTrue(any(i["code"] == "null_result_as_proof_of_no_effect" for i in issues))
        self.assertIn("未发现", issues[0]["source_sentence"])

    def test_cautionary_sentence_not_flagged(self):
        for text in [
            "必须强调：未发现差异不等于证明无效。",
            "不能把未检出差异写成证明无效，证据不足时应等待新研究。",
        ]:
            self.assertEqual(audit_wording(text), [], text)

    def test_clean_text_no_issue(self):
        self.assertEqual(audit_wording("甲方案使替代指标平均下降。"), [])


class CaseEvidenceChainTest(unittest.TestCase):
    def setUp(self):
        self.report = base_report()

    def test_one_independent_trial_for_ab7_ab8(self):
        self.assertEqual(self.report["stats"]["trial_count"], 1)
        trial = self.report["trials"][0]
        self.assertEqual(trial["trial_id"], "AB-7")
        self.assertEqual(trial["member_record_ids"], ["AB-7", "AB-8"])
        self.assertTrue(trial["has_follow_up_analysis"])

    def test_correction_propagates_status(self):
        status = {r["id"]: r["status"] for r in self.report["records"]}
        self.assertEqual(status["AB-7"], "corrected")
        self.assertTrue(any(e["to_status"] == "corrected" and e["caused_by"] == "AB-9"
                            for e in self.report["status_events"]))

    def test_g3_reopen_with_all_key_reasons(self):
        rev = review_by_id(self.report, "G-3")
        self.assertTrue(rev["needs_review"])
        self.assertEqual(rev["suggested_action"], "reopen")
        codes = check_codes(rev)
        self.assertIn("source_corrected", codes)
        self.assertIn("outcome_not_studied", codes)
        self.assertIn("surrogate_only_for_patient_outcome", codes)
        self.assertIn("population_scope_narrower", codes)

    def test_g3_trace_back_to_ab7_with_single_independent_study(self):
        rev = review_by_id(self.report, "G-3")
        self.assertIn("AB-7", rev["trace"]["member_record_ids"])
        self.assertEqual(rev["trace"]["independent_study_count"], 1)
        self.assertIn("AB-9", rev["trace"]["dossier_record_ids"])
        # 回查路径中必须有 G->AB-7 与 AB-9->AB-7
        pairs = {(p["from"], p["to"]) for p in rev["trace"]["paths"]}
        self.assertTrue(any(f == "G-3" and t == "AB-7" for f, t in pairs))
        self.assertIn(("AB-9", "AB-7"), pairs)

    def test_review_queue_contains_g3(self):
        self.assertEqual([q["guidance_id"] for q in self.report["re_review_queue"]], ["G-3"])

    def test_all_claim_sentences_verified(self):
        self.assertEqual(self.report["stats"]["unverified_source_sentences"], [])
        texts = {r["id"]: r["text"] for r in self.report["records"]}
        for c in self.report["claims"]:
            self.assertIn(c["source_sentence"], texts[c["record_id"]])

    def test_report_is_json_serializable(self):
        json.dumps(self.report, ensure_ascii=False)

    def test_deterministic_output(self):
        a = json.dumps(process_case(CASE), ensure_ascii=False, sort_keys=True)
        b = json.dumps(process_case(CASE), ensure_ascii=False, sort_keys=True)
        self.assertEqual(a, b)


class ChangeImpactTest(unittest.TestCase):
    def test_notice_and_followup_changes_impact_g3(self):
        rep = base_report()
        self.assertEqual(impacted_guidance_ids(rep, ["AB-9"]), ["G-3"])
        self.assertEqual(impacted_guidance_ids(rep, ["AB-8"]), ["G-3"])

    def test_retraction_escalates_g3(self):
        out = review_after_change(CASE, {"AB-7": "retracted"})
        self.assertEqual(out["impacted_guidance_ids"], ["G-3"])
        rev = review_by_id(out["updated_report"], "G-3")
        self.assertIn("source_retracted", check_codes(rev))

    def test_maintain_to_reopen_is_newly_flagged(self):
        case = {
            "case_id": "mini-clean",
            "records": [{
                "id": "S1", "kind": "abstract",
                "text": "在 50–75 岁确诊冠心病的患者中，丙方案与常规治疗相比使心肌梗死发生率下降。",
            }],
            "guidance": [{"id": "G1", "text": "50–75 岁确诊冠心病患者可使用丙方案降低心肌梗死风险。"}],
        }
        baseline = process_case(case)
        self.assertEqual(review_by_id(baseline, "G1")["suggested_action"], "maintain")
        out = review_after_change(case, {"S1": "retracted"})
        self.assertEqual(out["newly_needs_review"], ["G1"])
        self.assertEqual(out["decision_changes"][0]["old_action"], "maintain")
        self.assertEqual(out["decision_changes"][0]["new_action"], "reopen")

    def test_diff_detects_no_change(self):
        r1, r2 = base_report(), base_report()
        d = diff_reports(r1, r2)
        self.assertEqual(d["newly_needs_review"], [])
        self.assertEqual(d["decision_changes"], [])


class DecisionPatternsTest(unittest.TestCase):
    def _process(self, records, guidance):
        return process_case({"case_id": "t", "records": records, "guidance": guidance})

    def test_narrow_for_broader_population_and_dose(self):
        rep = self._process(
            [{
                "id": "N1", "kind": "abstract",
                "text": "在 18–40 岁轻度痤疮患者中，丁凝胶 0.1% 每日一次可减少皮损数量。",
            }],
            [{"id": "GN", "text": "各年龄段人群均可使用丁凝胶。"}],
        )
        rev = review_by_id(rep, "GN")
        self.assertEqual(rev["suggested_action"], "narrow")
        self.assertIn("population_scope_narrower", check_codes(rev))

    def test_null_only_support_reopens_and_keeps_inconclusive(self):
        rep = self._process(
            [{
                "id": "Z1", "kind": "abstract",
                "text": "在衰弱老人中，戊补充剂对认知评分的影响与安慰剂相比未发现显著差异。",
            }],
            [{"id": "GZ", "text": "推荐衰弱老人使用戊补充剂改善认知。"}],
        )
        rev = review_by_id(rep, "GZ")
        self.assertEqual(rev["suggested_action"], "reopen")
        self.assertIn("inconclusive_result_cannot_support_positive", check_codes(rev))
        # 任何主张都不能把它写成 no_effect_demonstrated
        self.assertTrue(all(c["direction"] != "no_effect_demonstrated" for c in rep["claims"]))

    def test_retracted_only_support_is_critical_reopen(self):
        rep = self._process(
            [
                {"id": "R1", "kind": "paper", "text": "己疫苗使65岁以上人群住院率下降。"},
                {"id": "R2", "kind": "retraction", "text": "撤稿声明：R1 被撤回。"},
            ],
            [{"id": "GR", "text": "建议65岁以上人群接种己疫苗降低住院风险。"}],
        )
        rev = review_by_id(rep, "GR")
        self.assertEqual(rev["suggested_action"], "reopen")
        self.assertEqual(rev["review_priority"], "critical")
        self.assertIn("source_retracted", check_codes(rev))

    def test_no_evidence_is_reopen(self):
        rep = self._process(
            [{"id": "X1", "kind": "abstract", "text": "这是一段与主题无关的植物学记录。"}],
            [{"id": "GX", "text": "建议市民使用火星草降低住院风险。"}],
        )
        rev = review_by_id(rep, "GX")
        self.assertEqual(rev["suggested_action"], "reopen")
        self.assertIn("no_traceable_evidence", check_codes(rev))

    def test_explicit_refs_used(self):
        rep = self._process(
            [{"id": "E1", "kind": "abstract", "text": "某干预在成人中改善血压。"}],
            [{"id": "GE", "text": "某建议。", "evidence_refs": ["E1"]}],
        )
        rev = review_by_id(rep, "GE")
        self.assertEqual(rev["trace"]["match_method"], "explicit_refs")
        self.assertIn("E1", rev["trace"]["seed_record_ids"])

    def test_short_intervention_name_matches_full_evidence_name(self):
        rep = self._process(
            [{"id": "K1", "kind": "abstract",
              "text": "在社区老人中，健身操方案与常规活动相比使跌倒发生率下降。"}],
            [{"id": "GK", "text": "推荐社区老人练习健身操以降低跌倒风险。"}],
        )
        rev = review_by_id(rep, "GK")
        self.assertNotIn("no_traceable_evidence", check_codes(rev))
        self.assertEqual(rev["suggested_action"], "maintain")
        self.assertIn("K1", rev["trace"]["member_record_ids"])


class CLITest(unittest.TestCase):
    def test_cli_writes_artifacts_and_check_exit_1(self):
        with tempfile.TemporaryDirectory() as td:
            code = cli_main([str(CASE_PATH), "--outdir", td, "--check"])
            self.assertEqual(code, 1)  # G-3 需要重审
            for name in ["evidence_report.json", "evidence_report.md",
                         "re_review_queue.json", "claims.json", "claims.csv",
                         "guidance_review.csv"]:
                self.assertTrue((Path(td) / name).exists(), name)
            data = json.loads((Path(td) / "evidence_report.json").read_text(encoding="utf-8"))
            self.assertEqual(data["case_id"], "evidence-7")

    def test_cli_status_override(self):
        with tempfile.TemporaryDirectory() as td:
            code = cli_main([str(CASE_PATH), "--outdir", td, "--status", "AB-7=retracted", "--no-write"])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
