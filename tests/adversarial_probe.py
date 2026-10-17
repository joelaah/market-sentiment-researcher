"""
Empirical Adversarial Probe & Stress Harness for NIFTY Relevance & FinBERT Analyzer.
Written by Challenger 1 (critic/specialist).
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from typing import Dict, Any, List

from src.relevance import NiftyRelevanceChecker, get_fallback_news_fixtures
from src.finbert_analyzer import FinBERTAnalyzer


def run_deceptive_keyword_tests(checker: NiftyRelevanceChecker) -> List[Dict[str, Any]]:
    deceptive_cases = [
        {
            "id": "DECEPTIVE_1_GADGET",
            "title": "A nifty new tech gadget in Silicon Valley revolutionizes photography",
            "summary": "Startups in California unveil nifty smartphone accessories featuring AI lens zoom.",
            "content": "A nifty new tech gadget in Silicon Valley revolutionizes photography. Startups in California unveil nifty smartphone accessories featuring AI lens zoom.",
            "expected_relevant": False,
            "description": "Adjective 'nifty' used for tech gadget"
        },
        {
            "id": "DECEPTIVE_2_PET_DOG",
            "title": "Meet Nifty, the rescued golden retriever who loves playing fetch in Seattle",
            "summary": "An adorable dog named Nifty won hearts at the local park agility championship.",
            "content": "Meet Nifty, the rescued golden retriever who loves playing fetch in Seattle. An adorable dog named Nifty won hearts at the local park agility championship.",
            "expected_relevant": False,
            "description": "Pet dog named 'Nifty'"
        },
        {
            "id": "DECEPTIVE_3_CLEANING_TIPS",
            "title": "Ten nifty tips for spring cleaning your house on a sunny weekend",
            "summary": "Use baking soda and vinegar for nifty kitchen hacks that remove stubborn grease.",
            "content": "Ten nifty tips for spring cleaning your house on a sunny weekend. Use baking soda and vinegar for nifty kitchen hacks that remove stubborn grease.",
            "expected_relevant": False,
            "description": "Adjective 'nifty' in household tips"
        },
        {
            "id": "DECEPTIVE_4_BSE_DISEASE",
            "title": "Outbreak of Bovine Spongiform Encephalopathy (BSE) confirmed by UK board",
            "summary": "Agricultural inspectors quarantine cattle farm in Yorkshire after BSE mad cow disease symptoms.",
            "content": "Outbreak of Bovine Spongiform Encephalopathy (BSE) confirmed by UK board. Agricultural inspectors quarantine cattle farm in Yorkshire after BSE mad cow disease symptoms.",
            "expected_relevant": False,
            "description": "BSE acronym for Bovine Spongiform Encephalopathy (mad cow disease)"
        },
        {
            "id": "DECEPTIVE_5_NSE_NIGERIA",
            "title": "Nigerian Stock Exchange (NSE) welcomes new telecommunications listing in Lagos",
            "summary": "Equities on the Nigerian Stock Exchange (NSE) rallied as domestic pension funds bought shares.",
            "content": "Nigerian Stock Exchange (NSE) welcomes new telecommunications listing in Lagos. Equities on the Nigerian Stock Exchange (NSE) rallied as domestic pension funds bought shares.",
            "expected_relevant": False,
            "description": "NSE acronym for Nigerian Stock Exchange"
        },
        {
            "id": "DECEPTIVE_6_TITAN_MOON",
            "title": "NASA discovers liquid methane lakes on Saturn's moon Titan",
            "summary": "Planetary scientists analyze radar data from the Cassini mission studying Titan atmosphere.",
            "content": "NASA discovers liquid methane lakes on Saturn's moon Titan. Planetary scientists analyze radar data from the Cassini mission studying Titan atmosphere.",
            "expected_relevant": False,
            "description": "Word 'Titan' referring to Saturn's moon"
        },
        {
            "id": "DECEPTIVE_7_POWER_GRID",
            "title": "Severe winter storm knocks out Texas power grid leaving millions freezing",
            "summary": "Freezing rain crippled power grid infrastructure and energy distribution across Texas.",
            "content": "Severe winter storm knocks out Texas power grid leaving millions freezing. Freezing rain crippled power grid infrastructure and energy distribution across Texas.",
            "expected_relevant": False,
            "description": "Word 'Power Grid' referring to electrical utilities"
        },
        {
            "id": "DECEPTIVE_8_BUDGET_HOTEL",
            "title": "European travel guide: How to find a charming budget hotel in Paris",
            "summary": "Travelers can save euros by booking boutique budget accommodation near the Seine river.",
            "content": "European travel guide: How to find a charming budget hotel in Paris. Travelers can save euros by booking boutique budget accommodation near the Seine river.",
            "expected_relevant": False,
            "description": "Word 'budget' in travel context"
        },
        {
            "id": "DECEPTIVE_9_ITC_SOLAR",
            "title": "US International Trade Commission (ITC) issues tariff ruling on solar panels",
            "summary": "The ITC determined that imported solar modules injured domestic manufacturing companies.",
            "content": "US International Trade Commission (ITC) issues tariff ruling on solar panels. The ITC determined that imported solar modules injured domestic manufacturing companies.",
            "expected_relevant": False,
            "description": "ITC acronym for International Trade Commission"
        },
        {
            "id": "DECEPTIVE_10_SBI_JAPAN",
            "title": "Tokyo financial firm SBI Holdings expands cryptocurrency exchange platform",
            "summary": "Japanese fintech group SBI Holdings reported increased crypto trading volumes in Tokyo.",
            "content": "Tokyo financial firm SBI Holdings expands cryptocurrency exchange platform. Japanese fintech group SBI Holdings reported increased crypto trading volumes in Tokyo.",
            "expected_relevant": False,
            "description": "SBI acronym for Japanese SBI Holdings"
        }
    ]

    results = []
    print("\n" + "=" * 80)
    print("1. RUNNING DECEPTIVE KEYWORD ADVERSARIAL PROBES")
    print("=" * 80)
    for case in deceptive_cases:
        res = checker.evaluate_article(case)
        is_rel = res["is_relevant"]
        passed = (is_rel == case["expected_relevant"])
        results.append({
            "id": case["id"],
            "description": case["description"],
            "expected": case["expected_relevant"],
            "actual": is_rel,
            "passed": passed,
            "score": res["relevance_score"],
            "entities": res["entity_matches"],
            "semantic_sim": res["semantic_similarity"],
            "sentiment": res["sentiment"],
            "materiality": res["materiality"],
            "reason": res["reason"]
        })
        status = "[PASS]" if passed else "[FAIL - FALSE POSITIVE]"
        print(f"{status:24} | {case['id']:20} | Rel: {is_rel!s:5} (Score: {res['relevance_score']:.3f}, SemSim: {res['semantic_similarity']*100:.1f}%) | Entities: {res['entity_matches']}")
        if not passed:
            print(f"   -> Title: '{case['title']}'")
            print(f"   -> Reason: {res['reason']}")
    return results


def run_genuine_indian_macro_tests(checker: NiftyRelevanceChecker) -> List[Dict[str, Any]]:
    genuine_cases = [
        {
            "id": "GENUINE_1_RBI_REPO",
            "title": "RBI cuts repo rate by 50 bps in surprise emergency meeting as GDP growth slows",
            "summary": "Governor Shaktikanta Das announced Monetary Policy Committee decision reducing benchmark rate to stimulate credit.",
            "content": "RBI cuts repo rate by 50 bps in surprise emergency meeting as GDP growth slows. Governor Shaktikanta Das announced Monetary Policy Committee decision reducing benchmark rate to stimulate credit.",
            "expected_relevant": True,
            "expected_mat": "HIGH"
        },
        {
            "id": "GENUINE_2_ADANI_FRAUD",
            "title": "Hindenburg Research report accuses Adani Group of massive corporate fraud and stock manipulation",
            "summary": "Adani Enterprises and constituent shares crash 20% on Dalal Street amid allegations of offshore shell money laundering.",
            "content": "Hindenburg Research report accuses Adani Group of massive corporate fraud and stock manipulation. Adani Enterprises and constituent shares crash 20% on Dalal Street amid allegations of offshore shell money laundering.",
            "expected_relevant": True,
            "expected_mat": "HIGH"
        },
        {
            "id": "GENUINE_3_BUDGET_TAX",
            "title": "Union Budget 2026: Finance Minister Nirmala Sitharaman raises STT on F&O and capital gains tax",
            "summary": "Stock market plunges as Dalal Street reacts to higher taxes on equity futures and short term capital gains.",
            "content": "Union Budget 2026: Finance Minister Nirmala Sitharaman raises STT on F&O and capital gains tax. Stock market plunges as Dalal Street reacts to higher taxes on equity futures and short term capital gains.",
            "expected_relevant": True,
            "expected_mat": "HIGH"
        },
        {
            "id": "GENUINE_4_NIFTY_RECORD",
            "title": "NIFTY 50 breaches 25,000 for first time as FII net inflows surge past Rs 12,000 crore",
            "summary": "Benchmark indices Sensex and Nifty surge on robust domestic corporate earnings and foreign institutional buying.",
            "content": "NIFTY 50 breaches 25,000 for first time as FII net inflows surge past Rs 12,000 crore. Benchmark indices Sensex and Nifty surge on robust domestic corporate earnings and foreign institutional buying.",
            "expected_relevant": True,
            "expected_mat": "HIGH"
        },
        {
            "id": "GENUINE_5_RELIANCE_TCS",
            "title": "Reliance Industries and TCS post record quarterly profit beats, lifting Dalal Street",
            "summary": "Heavyweights Reliance and TCS drive NSE trading volumes as operating margins expand in Q2 results.",
            "content": "Reliance Industries and TCS post record quarterly profit beats, lifting Dalal Street. Heavyweights Reliance and TCS drive NSE trading volumes as operating margins expand in Q2 results.",
            "expected_relevant": True,
            "expected_mat": "HIGH"
        }
    ]

    results = []
    print("\n" + "=" * 80)
    print("2. RUNNING GENUINE INDIAN MACRO SHOCK PROBES")
    print("=" * 80)
    for case in genuine_cases:
        res = checker.evaluate_article(case)
        is_rel = res["is_relevant"]
        mat = res["materiality"]
        rel_passed = (is_rel == case["expected_relevant"])
        mat_passed = (mat == case["expected_mat"])
        passed = rel_passed and mat_passed
        results.append({
            "id": case["id"],
            "expected_rel": case["expected_relevant"],
            "actual_rel": is_rel,
            "expected_mat": case["expected_mat"],
            "actual_mat": mat,
            "passed": passed,
            "score": res["relevance_score"],
            "entities": res["entity_matches"],
            "semantic_sim": res["semantic_similarity"],
            "sentiment": res["sentiment"]
        })
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:10} | {case['id']:20} | Rel: {is_rel!s:5} (Score: {res['relevance_score']:.3f}, SemSim: {res['semantic_similarity']*100:.1f}%) | Mat: {mat:6} (Expected: {case['expected_mat']}) | Entities: {res['entity_matches']}")
    return results


def run_international_macro_tests(checker: NiftyRelevanceChecker) -> List[Dict[str, Any]]:
    intl_cases = [
        {
            "id": "INTL_1_FED_RATE_CUT",
            "title": "Federal Reserve cuts US interest rates by 50 bps at FOMC meeting as Powell signals easing",
            "summary": "US central bank lowers federal funds rate to 4.75% as inflation slows toward 2% target.",
            "content": "Federal Reserve cuts US interest rates by 50 bps at FOMC meeting as Powell signals easing. US central bank lowers federal funds rate to 4.75% as inflation slows toward 2% target."
        },
        {
            "id": "INTL_2_OPEC_CRUDE_OIL",
            "title": "Crude oil plunges 8% as OPEC+ unexpectedly increases production quotas in Vienna",
            "summary": "Brent crude futures tumbled below $70 a barrel after Saudi Arabia and OPEC members agreed to boost oil output.",
            "content": "Crude oil plunges 8% as OPEC+ unexpectedly increases production quotas in Vienna. Brent crude futures tumbled below $70 a barrel after Saudi Arabia and OPEC members agreed to boost oil output."
        },
        {
            "id": "INTL_3_BOJ_YEN",
            "title": "Bank of Japan raises interest rates to 0.5%, unwinding global yen carry trades",
            "summary": "Governor Ueda signaled further rate increases in Tokyo, causing Japanese yen to strengthen against US dollar.",
            "content": "Bank of Japan raises interest rates to 0.5%, unwinding global yen carry trades. Governor Ueda signaled further rate increases in Tokyo, causing Japanese yen to strengthen against US dollar."
        },
        {
            "id": "INTL_4_SEATTLE_HOUSING",
            "title": "Seattle residential home prices rise 1.4% in local county regional market assessment",
            "summary": "Single family home inventory across Seattle and suburban King County registered slight gains as mortgage rates remained elevated.",
            "content": "Seattle residential home prices rise 1.4% in local county regional market assessment. Single family home inventory across Seattle and suburban King County registered slight gains as mortgage rates remained elevated."
        }
    ]

    results = []
    print("\n" + "=" * 80)
    print("3. RUNNING INTERNATIONAL MACRO STORIES PROBES")
    print("=" * 80)
    for case in intl_cases:
        res = checker.evaluate_article(case)
        results.append({
            "id": case["id"],
            "title": case["title"],
            "is_relevant": res["is_relevant"],
            "score": res["relevance_score"],
            "entities": res["entity_matches"],
            "semantic_sim": res["semantic_similarity"],
            "materiality": res["materiality"],
            "reason": res["reason"]
        })
        print(f"INTL PROBE | {case['id']:22} | Rel: {res['is_relevant']!s:5} (Score: {res['relevance_score']:.3f}, SemSim: {res['semantic_similarity']*100:.1f}%) | Mat: {res['materiality']:6} | Entities: {res['entity_matches']}")
        print(f"   -> Analysis: {res['reason']}")
    return results


def run_edge_case_tests(checker: NiftyRelevanceChecker, finbert: FinBERTAnalyzer) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("4. RUNNING EDGE CASE STRESS TESTS (LONG, EMPTY, UNICODE, EMOJIS)")
    print("=" * 80)
    results = []

    # 4.1 Empty article
    try:
        empty_res = checker.evaluate_article({"title": "", "summary": "", "content": ""})
        print(f"[PASS] Empty article handled: is_relevant={empty_res['is_relevant']}, score={empty_res['relevance_score']}")
        results.append({"test": "empty_article", "status": "PASS", "data": empty_res})
    except Exception as e:
        print(f"[FAIL] Empty article crashed: {e}")
        results.append({"test": "empty_article", "status": "FAIL", "error": str(e)})

    # 4.2 None fields
    try:
        none_res = checker.evaluate_article({"title": None, "summary": None, "content": None})
        print(f"[PASS] None fields article handled: is_relevant={none_res['is_relevant']}, score={none_res['relevance_score']}")
        results.append({"test": "none_fields", "status": "PASS", "data": none_res})
    except Exception as e:
        print(f"[FAIL] None fields article crashed: {e}")
        results.append({"test": "none_fields", "status": "FAIL", "error": str(e)})

    # 4.3 Whitespace only
    try:
        ws_res = checker.evaluate_article({"title": "   \n\t   ", "summary": "   ", "content": "   "})
        print(f"[PASS] Whitespace article handled: is_relevant={ws_res['is_relevant']}, score={ws_res['relevance_score']}")
        results.append({"test": "whitespace_only", "status": "PASS", "data": ws_res})
    except Exception as e:
        print(f"[FAIL] Whitespace article crashed: {e}")
        results.append({"test": "whitespace_only", "status": "FAIL", "error": str(e)})

    # 4.4 Extremely long article (>10,000 chars)
    long_text = ("The Reserve Bank of India (RBI) Monetary Policy Committee (MPC) announced repo rate decisions. "
                 "NIFTY 50 benchmark index and Dalal Street equities reacted to monetary policy and inflation. ") * 100
    try:
        t0 = time.time()
        long_res = checker.evaluate_article({
            "title": "NIFTY 50 and RBI Repo Rate Analysis Comprehensive Report",
            "summary": long_text[:500],
            "content": long_text
        })
        el = time.time() - t0
        print(f"[PASS] Long text ({len(long_text)} chars) handled in {el:.2f}s: is_relevant={long_res['is_relevant']}, FinBERT net={long_res['sentiment']['net_score']}")
        results.append({"test": "long_text_10k_chars", "status": "PASS", "elapsed": el, "data": long_res})
    except Exception as e:
        print(f"[FAIL] Long text crashed: {e}")
        results.append({"test": "long_text_10k_chars", "status": "FAIL", "error": str(e)})

    # 4.5 Broken unicode and null bytes
    unicode_text = "Nifty\ufffds rally \u201cAll-Time High\u201d \u2013 Dalal Street \x00\x01\x02 \ud83d\ude00 \ud83d\ude80"
    try:
        # Note: lone surrogates might fail python encoding if not handled
        safe_unicode = "Nifty\ufffds rally \u201cAll-Time High\u201d \u2013 Dalal Street \x00\x01\x02 \U0001F600 \U0001F680"
        uni_res = checker.evaluate_article({
            "title": safe_unicode,
            "summary": "Testing unicode characters and null bytes in headline",
            "content": safe_unicode
        })
        print(f"[PASS] Unicode and null bytes handled: is_relevant={uni_res['is_relevant']}")
        results.append({"test": "unicode_null_bytes", "status": "PASS", "data": uni_res})
    except Exception as e:
        print(f"[FAIL] Unicode text crashed: {e}")
        results.append({"test": "unicode_null_bytes", "status": "FAIL", "error": str(e)})

    # 4.6 Emojis only headline
    try:
        emoji_res = checker.evaluate_article({
            "title": "🚀🚀🚀🔥🔥🔥💰💰💰",
            "summary": "📈📈📈",
            "content": "🚀🚀🚀"
        })
        print(f"[PASS] Emojis only handled: is_relevant={emoji_res['is_relevant']}, sentiment={emoji_res['sentiment']['polarity']}")
        results.append({"test": "emoji_only", "status": "PASS", "data": emoji_res})
    except Exception as e:
        print(f"[FAIL] Emojis only crashed: {e}")
        results.append({"test": "emoji_only", "status": "FAIL", "error": str(e)})

    # 4.7 FinBERT extreme sentiment / sarcasm / negation
    fb_tests = [
        ("NIFTY crashes 15% in catastrophic Dalal Street bloodbath as financial contagion spreads", "negative"),
        ("TCS and Reliance achieve historic record-breaking explosive profit explosion with 100% margin surge", "positive"),
        ("This is not good news for the Indian banking sector as bad loans surge uncontrollably", "negative"),
        ("The company did not fail to exceed earnings expectations by a wide margin", "positive"),
        ("A total disaster, worst loss in company history, bankruptcy declared immediately", "negative"),
        ("Market closed flat with zero movement and negligible volume across all sectors", "neutral")
    ]
    print("\n--- FinBERT Polarity & Confidence Accuracy Probes ---")
    for text, expected_pol in fb_tests:
        sent = finbert.analyze(text)
        pol = sent["polarity"]
        conf = sent["confidence"]
        net = sent["net_score"]
        match = (pol == expected_pol)
        status = "[PASS]" if match else "[WARN/MISMATCH]"
        print(f"{status:16} | Text: '{text[:60]}...' -> Polarity: {pol} (conf: {conf:.2f}, net: {net:+.2f}) [Expected: {expected_pol}]")
        results.append({"test": "finbert_accuracy", "text": text, "expected": expected_pol, "actual": pol, "conf": conf, "net": net, "match": match})

    return results


def run_latency_and_batch_benchmarks(checker: NiftyRelevanceChecker, finbert: FinBERTAnalyzer) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("5. PERFORMANCE, LATENCY & BATCH BENCHMARKING")
    print("=" * 80)

    sample_texts = [
        "NIFTY 50 hits record high as RBI keeps repo rate unchanged and forecasts strong economic growth.",
        "Reliance Industries reports stellar earnings results with double digit growth across all units.",
        "US Federal Reserve signals upcoming interest rate cuts at Jackson Hole economic symposium.",
        "HDFC Bank and ICICI Bank lead banking sector gains on Dalal Street amid foreign inflows.",
        "European markets slip as tech shares fall following earnings downgrade in Frankfurt."
    ]

    # Benchmark individual FinBERT analyze
    latencies_fb = []
    for t in sample_texts:
        t0 = time.perf_counter()
        _ = finbert.analyze(t)
        latencies_fb.append(time.perf_counter() - t0)

    avg_fb_ms = (sum(latencies_fb) / len(latencies_fb)) * 1000
    print(f"FinBERT single analyze() latency: avg {avg_fb_ms:.1f}ms (min: {min(latencies_fb)*1000:.1f}ms, max: {max(latencies_fb)*1000:.1f}ms)")

    # Benchmark NiftyRelevanceChecker evaluate_article
    fixtures = get_fallback_news_fixtures()
    latencies_eval = []
    for art in fixtures:
        t0 = time.perf_counter()
        _ = checker.evaluate_article(art)
        latencies_eval.append(time.perf_counter() - t0)

    avg_eval_ms = (sum(latencies_eval) / len(latencies_eval)) * 1000
    print(f"NiftyRelevanceChecker evaluate_article() latency: avg {avg_eval_ms:.1f}ms (min: {min(latencies_eval)*1000:.1f}ms, max: {max(latencies_eval)*1000:.1f}ms)")

    # Calculate throughput for 110 articles (typical live feed size)
    est_110_sec = (avg_eval_ms * 110) / 1000
    print(f"Estimated time to evaluate 110 articles sequentially: {est_110_sec:.1f} seconds (~{est_110_sec/60:.2f} minutes)")

    # Benchmark batch_analyze implementation
    t0 = time.perf_counter()
    batch_res = finbert.batch_analyze(sample_texts)
    batch_total_s = time.perf_counter() - t0
    print(f"FinBERT batch_analyze() on 5 texts: {batch_total_s:.2f}s ({len(sample_texts)/batch_total_s:.1f} texts/sec)")

    return {
        "avg_finbert_ms": avg_fb_ms,
        "avg_evaluate_article_ms": avg_eval_ms,
        "est_110_articles_sec": est_110_sec,
        "batch_analyze_5_texts_sec": batch_total_s
    }


def main():
    print("================================================================================")
    print("     EMPIRICAL ADVERSARIAL STRESS SUITE: NIFTY RELEVANCE & FINBERT ANALYZER    ")
    print("================================================================================")

    checker = NiftyRelevanceChecker()
    finbert = checker._get_finbert()

    deceptive_res = run_deceptive_keyword_tests(checker)
    genuine_res = run_genuine_indian_macro_tests(checker)
    intl_res = run_international_macro_tests(checker)
    edge_res = run_edge_case_tests(checker, finbert)
    perf_res = run_latency_and_batch_benchmarks(checker, finbert)

    # Calculate summary statistics
    deceptive_failures = [r for r in deceptive_res if not r["passed"]]
    genuine_failures = [r for r in genuine_res if not r["passed"]]

    print("\n" + "=" * 80)
    print("                      ADVERSARIAL STRESS SUMMARY REPORT                         ")
    print("=" * 80)
    print(f"Deceptive Keyword Tests: {len(deceptive_res) - len(deceptive_failures)}/{len(deceptive_res)} Passed. ({len(deceptive_failures)} False Positives)")
    for f in deceptive_failures:
        print(f"  [X] FALSE POSITIVE: {f['id']} - {f['description']} (Entities: {f['entities']}, Score: {f['score']})")

    print(f"\nGenuine Indian Macro Tests: {len(genuine_res) - len(genuine_failures)}/{len(genuine_res)} Passed.")
    for f in genuine_failures:
        print(f"  [X] FAILURE: {f['id']} (Rel: {f['actual_rel']}, Mat: {f['actual_mat']})")

    print(f"\nBatch Throughput & Latency:")
    print(f"  * Per-article evaluation latency: {perf_res['avg_evaluate_article_ms']:.1f} ms")
    print(f"  * Projected 110-article feed evaluation time: {perf_res['est_110_articles_sec']:.1f} s")

    if deceptive_failures:
        print("\nGATE ASSESSMENT: VULNERABILITIES IDENTIFIED IN RELEVANCE FILTERING.")
    else:
        print("\nGATE ASSESSMENT: ALL ADVERSARIAL CHALLENGES SURVIVED.")


if __name__ == "__main__":
    main()
