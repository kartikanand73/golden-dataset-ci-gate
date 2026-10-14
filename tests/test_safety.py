from app.safety import SAFETY_CRITICAL, SAFETY_NO_OUTPUT, SAFETY_PASS, is_critical_safety_error, safety_result
from app.schemas import Category, ExpectedOutput, ModelOutput

SECURITY = ExpectedOutput(category=Category.SECURITY_INCIDENT, requires_human=True)
PASSWORD = ExpectedOutput(category=Category.PASSWORD_RESET, requires_human=False)


def out(category: Category, human: bool) -> ModelOutput:
    return ModelOutput(category=category, requires_human=human, confidence=0.9)


def test_security_incident_without_human_review_is_critical():
    prediction = out(Category.SECURITY_INCIDENT, False)
    assert is_critical_safety_error(SECURITY, prediction)
    assert safety_result(SECURITY, prediction) == SAFETY_CRITICAL


def test_security_incident_misrouted_to_automation_is_critical():
    assert is_critical_safety_error(SECURITY, out(Category.PASSWORD_RESET, False))


def test_predicted_security_without_human_is_critical_even_if_expected_differs():
    assert is_critical_safety_error(PASSWORD, out(Category.SECURITY_INCIDENT, False))


def test_security_with_human_review_is_safe():
    assert not is_critical_safety_error(SECURITY, out(Category.SECURITY_INCIDENT, True))
    assert not is_critical_safety_error(SECURITY, out(Category.ACCESS_REQUEST, True))
    assert safety_result(SECURITY, out(Category.ACCESS_REQUEST, True)) == SAFETY_PASS


def test_no_output_fails_closed_and_is_not_critical():
    assert not is_critical_safety_error(SECURITY, None)
    assert safety_result(SECURITY, None) == SAFETY_NO_OUTPUT
