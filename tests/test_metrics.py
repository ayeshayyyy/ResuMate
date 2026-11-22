from evaluation.metrics import evaluate, norm, match_doc, prf


def test_norm():
    assert norm("B.Tech.") == "b.tech" or norm("B.Tech.") == "b tech" or norm("B.Tech.") == "b.tech"
    assert norm("C++, ") == "c++"


def test_exact_vs_partial():
    gold, pred = ["Software Engineer"], ["Engineer"]
    assert match_doc(gold, pred, "exact")[0] == 0
    assert match_doc(gold, pred, "partial")[0] == 1


def test_prf_and_counts():
    gold = [{"Name": ["A B"], "Company": ["X", "Y"]}]
    pred = [{"Name": ["A B"], "Company": ["X", "Z"]}]
    t, err = evaluate(gold, pred, ["Name", "Company"])
    assert t["Name"]["f1"] == 1.0
    assert (t["Company"]["tp"], t["Company"]["fp"], t["Company"]["fn"]) == (1, 1, 1)
    assert t["micro"]["tp"] == 2 and t["micro"]["fp"] == 1 and t["micro"]["fn"] == 1
    assert err["Company"]["fp"] == [(0, "z")] and err["Company"]["fn"] == [(0, "y")]
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_duplicates_counted_once():
    t, _ = evaluate([{"Name": ["Ann"]}], [{"Name": ["Ann", "ann"]}], ["Name"])
    assert t["Name"]["fp"] == 0
