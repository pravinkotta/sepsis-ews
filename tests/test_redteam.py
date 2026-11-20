from src.redteam import main, run_all, surface_checks


def test_all_redteam_scenarios_pass():
    results = run_all()
    assert len(results) == 11
    failures = [r for r in results if not r.passed]
    assert not failures, failures


def test_tool_surface_checks_pass():
    results = surface_checks()
    assert results
    assert all(r.passed for r in results), results


def test_check_cli_exits_zero():
    assert main(["--check"]) == 0
