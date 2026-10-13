"""Defines the order and display labels of the five instruction roles and the three additional experiment 2 roles used on figure axes, plus a helper that joins a label onto one line. It reads and writes nothing."""

ROLE_ORDER = ("steward", "guardrail", "empathetic", "neutral", "devotion")

ROLE_LABEL = {"steward": "Patient +\nward priority", "guardrail": "Patient +\nstewardship",
              "empathetic": "Patient +\nharm avoidance", "neutral": "Neutral\nassignment",
              "devotion": "Strong\nadvocacy", "sx_patient_plain": "Matched:\npatient scope",
              "sx_ward_plain": "Matched:\nward scope",
              "devotion_precedence": "Strong advocacy +\nprecedence statement"}


def one_line(label):
    return label.replace("\n", " ")
