from datetime import datetime
from langchain_upstage import UpstageGroundednessCheck

current_date = datetime.now().strftime("%Y년 %m월 %d일")

def perform_groundedness_check(context, answer) -> str:
    groundedness_check = UpstageGroundednessCheck()
    response = groundedness_check.invoke({
        "context": context,
        "answer": answer
    })

    # 검증완료 (grounded)
    # 검증실패 (notGrounded)
    # 검증불가 (notSure)

    return response

grounded_result_mapping = {
    "grounded": ("검증완료", "green"),
    "notGrounded": ("검증실패", "red"),
    "notSure": ("검증불가", "orange")
}