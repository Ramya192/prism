# domains/payroll_hr/payroll/agents/retriever_agent.py
# Subclasses core/rag/base_retriever_agent.py. The four class attributes
# below reproduce this domain's exact original _hyde()/_generate_variants()
# prompt text, character for character.

from core.rag.base_retriever_agent import BaseRetrieverAgent
from domains.payroll_hr.payroll.settings import Settings


class RetrieverAgent(BaseRetrieverAgent):
    PERSONA = "a payroll compliance analyst"
    HYDE_DOC_PHRASE = "payslip/payroll-register"
    HYDE_INSTRUCTION = (
        "Be specific with amounts, pay categories (gross pay, deductions, tax\n"
        "withholding, overtime), and dates."
    )
    VARIANT_DOC_PHRASE = "a payslip or payroll register PDF"

    def __init__(self):
        super().__init__(Settings)
